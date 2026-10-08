"""
Manifest versionati di CASIA-FASD e CelebA-Spoof e controllo delle sovrapposizioni fra gli split (M2, seconda tranche).
Stesso schema della tranche NUAA (scripts/nuaa_manifests.py, da cui importa lettura dei CSV, scansione del contenuto
pubblico e hash dei frame derivati), con un motore di regole generico sulle dimensioni di ciascun dataset.

Sottocomandi (sempre con --dataset casia_fasd oppure celeba_spoof):
  build          legge le copie locali (CASIA: frame color del mirror; CelebA: righe del parquet dello shard 0), le
                 cartelle preparate, le cartelle di valutazione dell'app (300 immagini, congelate: solo confronto) e la
                 cache di addestramento; calcola SHA-256 e byte di ogni immagine (sola lettura) e scrive in
                 <out>/<dataset>/manifests/ i manifest per immagine:
                   CASIA: mirror_train, mirror_test, prepared_training, prepared_validation, eval, cnn_train, cnn_es,
                          cnn_test, pooled_train, pooled_es, pooled_test
                   CelebA: parquet (6.678 righe con immagine), prepared_training, prepared_validation, eval, cnn_train,
                          cnn_es, cnn_test, pooled_train, pooled_es, pooled_test
                 Gli split delle CNN sono ricostruiti con la logica del commit 0ac7143 di scripts/train_cnn.py (prima
                 versione versionata: GroupShuffleSplit per l'official, 80/20 stratificato più 10 % stratificato per il
                 pooled) e verificati sui conteggi delle schede, sugli array y/s della cache e sullo SHA-256 dei .h5
                 (scheda, models/weights/manifest.json e file). I frame derivati della cache sono solo hashati.
                 Scrive <dataset>_manifests_summary.json (riservato) e <dataset>_manifests_aggregate.json (pubblico,
                 senza identificativi, scansionato). Codice 1 se un controllo fallisce.
  check-overlap  regole dichiarate (delivery/m2/rgb/<dataset>_overlap_rules.json): righe e SHA-256 attesi di ogni
                 manifest, validazione stretta di ogni riga contro insiemi chiusi (nessun ripiego), poi le regole
                 (disgiunzione, inclusione, intersezioni dichiarate, duplicati, derivazioni, coerenza per chiave,
                 dichiarazioni non verificabili). Rapporti riservato e pubblico. Con --self-test ripete il controllo su
                 copie guaste dei manifest reali (righe e SHA-256 ridichiarati) e pretende le regole attese.
  recount        ricalcola dai soli manifest il blocco ricontabile dell'aggregato pubblico (codice 1 se differisce).
  reconstruct    ricostruisce dai soli manifest gli split official e pooled (scikit-learn, nessuna immagine) e li
                 confronta con quelli dichiarati; una deriva della libreria è segnalata con un errore esplicito.
  report         rapporto breve in inglese (privato e pubblico) dal modello delivery/m2/rgb/report.md.tmpl.

Nessuna inferenza, nessun addestramento, nessun modello caricato. Non scrive mai in data, models, results/c1,
results/eval, results/m1 o results/m2 (tranche NUAA). Dipendenze: numpy (tutti i sottocomandi); scikit-learn per build
e reconstruct; pyarrow per build di CelebA-Spoof.

Uso tipico:
  python scripts/rgb_manifests.py build --dataset casia_fasd
  python scripts/rgb_manifests.py check-overlap --dataset casia_fasd --self-test
  python scripts/rgb_manifests.py report --dataset casia_fasd
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import numpy as np  # noqa: E402
    import nuaa_manifests as nm  # noqa: E402  (importa anche select_threshold e src/ al percorso)
except ModuleNotFoundError as _e:   # dipendenza mancante: messaggio chiaro, mai un traceback
    print(f"ERROR: missing Python package {_e.name!r}: install the requirements (numpy for every subcommand, scikit-learn "
          "for build and reconstruct, pyarrow for the CelebA-Spoof build)", file=sys.stderr)
    sys.exit(1)
from nuaa_manifests import SHA_RE, as_list, examples_of, frame_hashes, plural, public_problems, read_csv  # noqa: E402
from select_threshold import fail, is_under, load_json, now, rel, relpath, repo_state, sha256, spread  # noqa: E402
from tesi_app.paths import DATA, RESULTS, WEIGHTS  # noqa: E402

SEED = 42
PER_CLASS = 1500
EVAL_PER_CLASS = 150
ARCHS = ["LivenessNet", "AttackNetV1", "AttackNetV2_1", "AttackNetV2_2"]
POOLED_ARCHS = ["LivenessNet", "AttackNetV2_2"]       # pooled solo per queste due architetture (schede esistenti)
DATASETS = ("casia_fasd", "celeba_spoof")
OUT_DIR = RESULTS / "m2_rgb"
RULES_DIR = ROOT / "delivery" / "m2" / "rgb"
REPORT_TMPL = RULES_DIR / "report.md.tmpl"
SPLIT_LOGIC_COMMIT = "0ac7143"                        # prima versione versionata di scripts/train_cnn.py
BUILT_WITH = {"scikit-learn": "1.3.2", "numpy": "1.26.4"}   # versioni con cui sono stati scritti i manifest dichiarati
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
SEVERITIES = ("must", "report")
NOT_CHECKABLE = "not checkable"
# valori di val_split delle schede (in italiano) -> inglese, solo per i rapporti
VAL_SPLIT_EN = {"per soggetto": "by subject", "casuale": "random", "casuale per immagine": "random per image"}


def rules_path(ds: str) -> Path:
    return RULES_DIR / f"{ds}_overlap_rules.json"


def ds_out(out: Path, ds: str) -> Path:
    return Path(out) / ds


# ============================================================================ CASIA-FASD: nomi, campi, validazione

CASIA_MIRROR = "akahana/anti-spoofing-casiafasd"
CASIA_ARCHIVE = DATA / "raw" / "casiafasd.tar.gz"
CASIA_HF_META = DATA / "raw" / ".cache" / "huggingface" / "download" / "casiafasd.tar.gz.metadata"
CASIA_RAW = DATA / "raw" / "casia"
CASIA_IMAGES = DATA / "CASIA_FASD" / "images"
CASIA_EVAL = DATA / "eval" / "casia_fasd"
CASIA_CACHE = DATA / "processed" / f"casia_fasd_{PER_CLASS}.npz"
CASIA_SPLIT_DIR = {"train": "train_img/train_img", "test": "test_img/test_img"}      # split del mirror -> cartella
CASIA_TAG = {"train": "training", "test": "validation"}                                # split del mirror -> cartelle preparate
CASIA_SUBJECTS = {"train": range(1, 21), "test": range(1, 31)}                        # numerazione locale allo split (copia)
CASIA_VIDEOS = ("1", "2", "3", "4", "5", "6", "7", "8", "HR_1", "HR_2", "HR_3", "HR_4")
CASIA_REAL_VIDEOS = ("1", "2", "HR_1")    # verificato sulla copia: sempre `real`; gli altri sempre `fake`
CASIA_COLUMNS = ["key", "label", "mirror_split", "subject", "subject_key", "video", "video_key", "frame", "sha256", "bytes"]
# nome del mirror: <soggetto>_<video>.avi_<frame>_<real|fake>.jpg ; chiave: <split del mirror>/<nome>
CASIA_NAME_RE = re.compile(r"^([1-9]\d?)_((?:HR_)?[1-8])\.avi_(0|[1-9]\d*)_(real|fake)\.jpg$")
CASIA_KEY_RE = re.compile(r"^(train|test)/(.+)$")


def casia_fields(key: str) -> dict:
    """Campi di una chiave CASIA `<train|test>/<nome del mirror>`; ValueError se la chiave non ha questa forma o se un
    campo è fuori dagli insiemi chiusi verificati sulla copia (soggetti 1–20 / 1–30, 12 token video, classe del token)."""
    m = CASIA_KEY_RE.match(key or "")
    if not m:
        raise ValueError(f"not a CASIA key of the form <train|test>/<name>: {key!r}")
    split, name = m.groups()
    n = CASIA_NAME_RE.match(name)
    if not n:
        raise ValueError(f"not a CASIA mirror name <subject>_<video>.avi_<frame>_<real|fake>.jpg: {key!r}")
    subject, video, frame, cls = n.groups()
    if int(subject) not in CASIA_SUBJECTS[split]:
        raise ValueError(f"subject {subject} outside the {split} numbering 1–{CASIA_SUBJECTS[split][-1]}: {key!r}")
    if video not in CASIA_VIDEOS:
        raise ValueError(f"video token {video!r} not in {list(CASIA_VIDEOS)}: {key!r}")
    if (cls == "real") != (video in CASIA_REAL_VIDEOS):
        raise ValueError(f"class suffix {cls!r} not consistent with video token {video!r} (verified on the copy): {key!r}")
    return {"label": "0" if cls == "real" else "1", "mirror_split": split, "subject": subject, "subject_key": f"{split}:{subject}",
            "video": video, "video_key": f"{split}:{subject}:{video}", "frame": frame}


def casia_row(split: str, name: str, sha: str, size: int) -> dict:
    key = f"{split}/{name}"
    return {"key": key, **casia_fields(key), "sha256": sha, "bytes": str(size)}


# ============================================================================ CelebA-Spoof: righe del parquet

CELEBA_MIRROR = "nguyenkhoa/celeba-spoof-for-face-antispoofing-test"
CELEBA_SHARD = "test-00000-of-00010.parquet"
CELEBA_PARQUET = DATA / "raw" / "celeba_spoof" / "data" / CELEBA_SHARD
CELEBA_HF_META = DATA / "raw" / "celeba_spoof" / ".cache" / "huggingface" / "download" / "data" / f"{CELEBA_SHARD}.metadata"
CELEBA_IMAGES = DATA / "CelebA_Spoof" / "images"
CELEBA_EVAL = DATA / "eval" / "celeba_spoof"
CELEBA_CACHE = DATA / "processed" / f"celeba_spoof_{PER_CLASS}.npz"
CELEBA_ROWS = 6717                                   # righe dello shard 0 (verificato da build)
CELEBA_COLUMNS = ["key", "label", "label_name", "row", "official_split", "sha256", "bytes"]
CELEBA_LABEL_NAME = {"0": "live", "1": "spoof"}
CELEBA_KEY_RE = re.compile(r"^celeba_(\d{5})$")      # chiave = nome del file preparato senza estensione
CELEBA_N_TRAINING, CELEBA_N_VALIDATION_EXTRA = 1500, 300   # regola delle cartelle preparate (sez. 2.2 dell'inventario)


def celeba_fields(key: str) -> dict:
    m = CELEBA_KEY_RE.match(key or "")
    if not m:
        raise ValueError(f"not a CelebA-Spoof key of the form celeba_<5-digit row>: {key!r}")
    row = int(m.group(1))
    if row >= CELEBA_ROWS:
        raise ValueError(f"row {row} outside the shard (0–{CELEBA_ROWS - 1}): {key!r}")
    return {"row": str(row), "official_split": "test"}


def celeba_row(i: int, label: int, sha: str, size: int) -> dict:
    return {"key": f"celeba_{i:05d}", "label": str(label), "label_name": CELEBA_LABEL_NAME[str(label)], "row": str(i),
            "official_split": "test", "sha256": sha, "bytes": str(size)}


# ============================================================================ specifica per dataset

SPEC = {
    "casia_fasd": {
        "title": "CASIA-FASD", "columns": CASIA_COLUMNS,
        "dimensions": {"key": "key", "hash": "sha256", "subject": "subject_key", "video": "video_key", "local_subject": "subject"},
        "manifests": {
            "mirror_train": ("casia_mirror_train.csv", "all colour frames of `train_img` of the mirror (official training split by protocol [DOC-EXT])"),
            "mirror_test": ("casia_mirror_test.csv", "all colour frames of `test_img` of the mirror (official test split by protocol [DOC-EXT])"),
            "prepared_training": ("casia_prepared_training.csv", "prepared folders `{bonafide,attack}_training` (hard links to `train_img`)"),
            "prepared_validation": ("casia_prepared_validation.csv", "prepared folders `{bonafide,attack}_validation` (hard links to `test_img`; never used as validation)"),
            "eval": ("casia_eval.csv", "the 300 images of the application evaluation set `data/eval/casia_fasd` (frozen, compared only)"),
            "cnn_train": ("casia_cnn_train.csv", "images that fitted the weights of the four CASIA-FASD official CNNs"),
            "cnn_es": ("casia_cnn_early_stopping.csv", "validation images of the four official CNNs (early stopping on val_loss, by subject)"),
            "cnn_test": ("casia_cnn_test.csv", "test images of the official checkpoint cards (spread(..., 1500) per class of `test_img`)"),
            "pooled_train": ("casia_pooled_train.csv", "images that fitted the weights of the two CASIA-FASD pooled CNNs"),
            "pooled_es": ("casia_pooled_early_stopping.csv", "validation images of the two pooled CNNs (random, stratified)"),
            "pooled_test": ("casia_pooled_test.csv", "test images of the pooled checkpoint cards (random 20 % of the pooled frames)"),
        },
    },
    "celeba_spoof": {
        "title": "CelebA-Spoof", "columns": CELEBA_COLUMNS,
        "dimensions": {"key": "key", "hash": "sha256"},
        "manifests": {
            "parquet": ("celeba_parquet.csv", "all rows with image bytes of shard 0 of the mirror (official test split; 39 rows without image excluded)"),
            "prepared_training": ("celeba_prepared_training.csv", "prepared folders `{bonafide,attack}_training` (files byte-identical to the parquet PNG bytes)"),
            "prepared_validation": ("celeba_prepared_validation.csv", "prepared folders `{bonafide,attack}_validation` (byte-identical files; the 300 evaluation images included)"),
            "eval": ("celeba_eval.csv", "the 300 images of the application evaluation set `data/eval/celeba_spoof` (frozen, compared only)"),
            "cnn_train": ("celeba_cnn_train.csv", "images that fitted the weights of the four CelebA-Spoof \"official\" CNNs (our per-image split inside the official test)"),
            "cnn_es": ("celeba_cnn_early_stopping.csv", "validation images of the four \"official\" CNNs (per image)"),
            "cnn_test": ("celeba_cnn_test.csv", "test images of the \"official\" checkpoint cards (the prepared `_validation` folders)"),
            "pooled_train": ("celeba_pooled_train.csv", "images that fitted the weights of the two CelebA-Spoof pooled CNNs"),
            "pooled_es": ("celeba_pooled_early_stopping.csv", "validation images of the two pooled CNNs (random, stratified)"),
            "pooled_test": ("celeba_pooled_test.csv", "test images of the pooled checkpoint cards (random 20 % of the pooled images)"),
        },
    },
}

# dichiarazioni sulla fonte (stesso testo in naming_schema.md, README.md e report.md.tmpl di delivery/m2/rgb)
SOURCE_DECLARATIONS = {
    "casia_fasd": (
        ("Original source", "CASIA Face Anti-Spoofing Database (CASIA-FASD), Institute of Automation, Chinese Academy of Sciences; "
                            "reference publication: Z. Zhang, J. Yan, S. Liu, Z. Lei, D. Yi, S. Z. Li, \"A face antispoofing database "
                            "with diverse attacks\", ICB 2012."),
        ("Documented [DOC-EXT]", "50 subjects, 20 in the training set and 30 in the test set, disjoint by protocol; 12 videos per "
                                 "subject (genuine, warped photo, cut photo and video replay, each at low, normal and high quality), from "
                                 "the reference publication; the mapping of the video tokens to attack type and quality is not in the "
                                 "mirror and is not applied: the manifests carry the token as an opaque code."),
        ("Verified on the copy", "only extracted JPEG frames, without videos, README or split lists: `train_img` 1,655 and `test_img` "
                                 "2,408 colour frames named `<subject>_<video>.avi_<frame>_<real|fake>.jpg`; subject numbers 1–20 in "
                                 "`train_img` and 1–30 in `test_img` (the numbering is local to the split: the same numbers occur in "
                                 "both); 12 video tokens per subject, none missing; tokens 1, 2 and HR_1 always `real`, the others "
                                 "always `fake`."),
        ("Not verified", "that the subjects of `train_img` and of `test_img` are different persons; the attack type and quality of "
                         "each token; how the frames were extracted and cropped; the origin of the `depth/` maps (not used); the "
                         "licence of the mirror."),
        ("Subjects and videos", "a subject is keyed `<mirror split>:<number>` and a video `<mirror split>:<number>:<token>`; a number "
                                "alone is never compared across splits as an identity. No identity is inferred from image content."),
    ),
    "celeba_spoof": (
        ("Original source", "CelebA-Spoof; reference publication: Y. Zhang, Z. Yin, Y. Li, G. Yin, J. Yan, J. Shao, Z. Liu, "
                            "\"CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations\", ECCV 2020; official "
                            "repository `ZhangYuanhan-AI/CelebA-Spoof`."),
        ("Documented [DOC-EXT]", "the original release organises the images by identity, with official training and test splits and "
                                 "annotations of attack type, illumination, environment, sensor and face attributes."),
        ("Verified on the copy", "shard 0 of 10 of the test split of the mirror: one parquet file with the columns `cropped_image` "
                                 "(`bytes`, `path`), `labels` (0 live, 1 spoof) and `labelNames`; 6,717 rows (2,030 live, 4,687 "
                                 "spoof), 39 of them without image bytes; `path` empty in every row; every image is a PNG; the prepared "
                                 "files are byte-identical to the PNG bytes of their rows (with a `.jpg` extension)."),
        ("Not verified", "the identity, attack type and acquisition conditions of any row; the order of the rows; how the faces "
                         "were cropped; how the shard relates to the whole official test split; the procedure that converted the "
                         "parquet into the prepared folders (a rule that reproduces them exactly is checked; the procedure itself is "
                         "not versioned); the licence of the mirror."),
        ("Identities", "none: the copy has no identity, subject or video field, so no split can be checked for identity overlap; "
                       "no identity is inferred from row order, content or similarity."),
    ),
}

# limiti e confronti sostenuti (stesso testo in README.md e nel rapporto)
LIMITS = {
    "casia_fasd": (
        "The copy has only extracted frames: no original video, README or official split list. That `train_img` and `test_img` "
        "hold different persons is documented by the protocol [DOC-EXT] and cannot be verified on the copy, because subject "
        "numbers are local to the split.",
        "Frames of the same video are not independent samples; every split of this tranche keeps the frames of a video together "
        "except the pooled one.",
        "The pooled checkpoints use a per-frame random split: all 50 subjects and 444 of the 600 videos are on both sides, and "
        "235 of the 300 evaluation images were used to fit or validate them.",
        "The training run did not save its file list: the splits are rebuilt with the logic of commit `0ac7143` and verified "
        "against the card counts and the label and subject arrays of the training cache; the card counts alone do not determine "
        "the four held-out subjects (75 combinations of 4 of the 20 training subjects give 316 images), so the official split "
        "relies on `GroupShuffleSplit` with seed 42. The training cache is linked to the files by sampling order, not image by image.",
        "317 test attack frames are not in the official test (spread(..., 1500) per class); the `depth/` maps are not used.",
    ),
    "celeba_spoof": (
        "The \"official\" checkpoints are not trained on the official CelebA-Spoof protocol: fitting, early stopping and test images "
        "all come from shard 0 of the official test split, divided by us per image.",
        "No identity is available: fitting, early-stopping and test images are not guaranteed to be identity-disjoint (overlap can "
        "neither be excluded nor measured; declared as not checkable); test results say nothing about unseen subjects.",
        "The early-stopping split is rebuilt with the first versioned logic (`GroupShuffleSplit` on file names, one image per "
        "group); a stratified split gives the same totals 2,400 / 600 (with 300 / 300 per class in the early stopping instead "
        "of 295 / 305). Which of the two the training run used is not verified. The "
        "`val_split` field of the four \"official\" cards was 'by subject' in commit `0ac7143` and was rewritten as 'random per "
        "image' in commit `4d11e67`; with the logic of `0ac7143` the first value corresponds to `GroupShuffleSplit` on groups of "
        "one image.",
        "The procedure that converted the parquet into the prepared folders is not versioned; a rule that reproduces their row "
        "sets exactly is checked. 39 rows without image bytes are excluded; 71 live and 2,707 spoof rows are not used.",
        "One pair of rows has byte-identical images (exact duplicate, both bona fide); near-duplicates are not searched.",
        "The pooled checkpoints use a random split of the same images: 244 of the 300 evaluation images were used to fit or "
        "validate them.",
    ),
}
SUPPORTED = {
    "casia_fasd": (
        "Supported: a descriptive comparison of the four official CNNs trained on the same split (fitting and early stopping "
        "subject-disjoint: verified on the reconstructed split, recorded as 'by subject' in the cards; test on `test_img`, "
        "subject-disjoint by protocol, documented), single seed, without an uncertainty estimate.",
        "Not supported: any claim on the pooled checkpoints evaluated on the evaluation set or on `test_img` (their training "
        "includes evaluation images, subjects and videos of the test side); a claim of subject disjointness between `train_img` "
        "and `test_img` as verified.",
    ),
    "celeba_spoof": (
        "Supported: a descriptive, within-shard comparison of the four \"official\" CNNs on the same per-image split, declared as "
        "not guaranteed to be identity-disjoint; CelebA-Spoof as a target dataset for models trained elsewhere, declared as shard 0 of the "
        "official test without identities.",
        "Not supported: conclusions on unseen subjects; comparisons with results on the official CelebA-Spoof protocol; any claim "
        "on the pooled checkpoints evaluated on the evaluation set.",
    ),
}


# ============================================================================ validazione stretta delle righe

def row_problems(ds: str, rows) -> list:
    """Ogni campo di ogni riga contro gli insiemi chiusi e contro i campi derivati dalla chiave, come testo esatto
    (nessun ripiego: valori vuoti, spazi, zeri iniziali o valori fuori dominio sono errori)."""
    bad = defaultdict(list)
    cols = SPEC[ds]["columns"]
    for r in rows:
        k = r.get("key")
        extra = sorted(c for c in r if c not in cols)
        if extra:
            bad[f"unexpected columns {extra}"].append(str(k))
        try:
            f = casia_fields(k) if ds == "casia_fasd" else celeba_fields(k)
        except ValueError:
            bad["keys outside the closed format of the dataset"].append(str(k)); continue
        if ds == "celeba_spoof":
            if r.get("label") not in CELEBA_LABEL_NAME:
                bad["label values other than 0/1"].append(k)
            elif r.get("label_name") != CELEBA_LABEL_NAME[r["label"]]:
                bad["label_name values not consistent with label (0 live, 1 spoof)"].append(k)
        for c, v in f.items():
            if r.get(c) != v:
                bad[f"{c} values not consistent with the key"].append(k)
        if not SHA_RE.match(r.get("sha256") or ""):
            bad["malformed sha256 values"].append(k)
        if not re.fullmatch(r"[1-9]\d*", r.get("bytes") or ""):
            bad["malformed bytes values (positive integer without leading zeros)"].append(k)
    return [f"{len(v)} {what} (e.g. {v[:3]})" for what, v in bad.items()]


def canonical_text(ds: str, rows) -> str:
    """Testo canonico di un manifest, come lo scrive write_manifest: intestazione del dataset, righe ordinate per chiave
    (ordinamento stabile), fine riga LF, nessun BOM."""
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=SPEC[ds]["columns"], lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: r["key"]))
    return buf.getvalue()


def load_manifest(ds: str, p: Path):
    """Righe di un manifest: intestazione esattamente quella del dataset, righe valide, file in forma canonica (byte
    uguali a quelli che scriverebbe write_manifest: LF, nessun BOM, righe ordinate per chiave, nessuna riga vuota)."""
    rows = read_csv(p, SPEC[ds]["columns"])
    with open(p, encoding="utf-8-sig", newline="") as fh:
        header = next(csv.reader(fh), [])
    if header != SPEC[ds]["columns"]:
        raise ManifestError(f"{relpath(p)}: header {header} differs from {SPEC[ds]['columns']}")
    problems = row_problems(ds, rows)
    if problems:
        raise ManifestError(f"{relpath(p)}: row validation failed: " + "; ".join(problems))
    raw = Path(p).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw or raw != canonical_text(ds, rows).encode("utf-8"):
        why = ("byte-order mark" if raw.startswith(b"\xef\xbb\xbf") else "CR line endings" if b"\r" in raw
               else "rows not sorted by key, blank lines or non-canonical quoting")
        raise ManifestError(f"{relpath(p)}: not in canonical form ({why}): manifests must be byte-identical to the output of "
                            "write_manifest (LF, no BOM, rows sorted by key)")
    return rows


class ManifestError(Exception):
    """Righe non valide: check-overlap le registra nel rapporto prima di uscire con codice 1."""


def write_manifest(ds: str, p: Path, rows) -> str:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SPEC[ds]["columns"], lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["key"]))
    return sha256(p)


def write_rows(ds: str, p: Path, rows) -> None:
    """Righe nell'ordine dato (per le copie guaste del self-test)."""
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SPEC[ds]["columns"], lineterminator="\n", extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


# ============================================================================ conteggi

def counts(ds: str, rows, private: bool) -> dict:
    lab = Counter(str(r["label"]) for r in rows)
    out = {"n": len(rows), "n_bona_fide": lab["0"], "n_attack": lab["1"]}
    if ds == "casia_fasd":
        out["n_subjects"] = len({r["subject_key"] for r in rows})
        out["n_videos"] = len({r["video_key"] for r in rows})
        by = defaultdict(lambda: {"bona_fide": 0, "attack": 0})
        for r in rows:
            by[r["mirror_split"]]["attack" if r["label"] == "1" else "bona_fide"] += 1
        out["by_mirror_split"] = {k: by[k] for k in sorted(by)}
        out["by_video_token"] = dict(sorted(Counter(r["video"] for r in rows).items(), key=lambda kv: CASIA_VIDEOS.index(kv[0])))
        if private:
            out["by_subject"] = dict(sorted(Counter(r["subject_key"] for r in rows).items()))
    return out


def recount_block(ds: str, m: dict) -> dict:
    """Conteggi dell'aggregato che si ricalcolano dai soli manifest: per manifest, esclusioni del campionamento,
    sovrapposizioni dichiarate dei pooled, duplicati esatti per hash. Nessun identificativo."""
    keys = {n: {r["key"] for r in rows} for n, rows in m.items()}
    trainval = keys["pooled_train"] | keys["pooled_es"]
    block = {"manifest_counts": {n: counts(ds, rows, private=False) for n, rows in m.items()}}
    if ds == "casia_fasd":
        sampled = keys["cnn_train"] | keys["cnn_es"]
        block["exclusions"] = {"mirror_train_not_sampled": counts(ds, [r for r in m["mirror_train"] if r["key"] not in sampled], False),
                               "mirror_test_not_sampled": counts(ds, [r for r in m["mirror_test"] if r["key"] not in keys["cnn_test"]], False)}
        side = lambda names, dim: {r[dim] for n in names for r in m[n]}   # noqa: E731
        block["pooled_overlap"] = {
            "eval_images_in_pooled_fit_or_validation": len(keys["eval"] & trainval),
            "subjects_on_both_sides": len(side(("pooled_train", "pooled_es"), "subject_key") & side(("pooled_test",), "subject_key")),
            "subjects_total": len(side(("pooled_train", "pooled_es", "pooled_test"), "subject_key")),
            "videos_on_both_sides": len(side(("pooled_train", "pooled_es"), "video_key") & side(("pooled_test",), "video_key")),
            "videos_total": len(side(("pooled_train", "pooled_es", "pooled_test"), "video_key")),
            "fit_validation_shared_subjects_official": len({r["subject_key"] for r in m["cnn_train"]} & {r["subject_key"] for r in m["cnn_es"]})}
    else:
        used = keys["prepared_training"] | keys["prepared_validation"]
        block["exclusions"] = {"parquet_rows_not_used": counts(ds, [r for r in m["parquet"] if r["key"] not in used], False)}
        block["pooled_overlap"] = {"eval_images_in_pooled_fit_or_validation": len(keys["eval"] & trainval)}
    block["exact_duplicates_by_hash"] = {n: sum(1 for c in Counter(r["sha256"] for r in rows).values() if c > 1) for n, rows in m.items()}
    return json.loads(json.dumps(block))


# ============================================================================ motore delle regole

KINDS = ("disjoint", "subset", "intersection", "no_internal_duplicates", "internal_duplicates", "partition", "equals_spread",
         "spread_partition", "celeba_conversion", "key_consistency", "counts", "declared_not_checkable")
KEY_ONLY = ("partition", "equals_spread", "spread_partition", "celeba_conversion", "key_consistency", "counts")


def count_fields(ds: str) -> tuple:
    """Conteggi dichiarati per manifest nelle regole `counts` (gli stessi dell'aggregato pubblico)."""
    return ("n", "n_bona_fide", "n_attack", "n_subjects", "n_videos") if ds == "casia_fasd" else ("n", "n_bona_fide", "n_attack")


def values(ds: str, rows, dim: str):
    col = SPEC[ds]["dimensions"].get(dim)
    if col is None:
        raise ValueError(f"unknown dimension {dim!r} for {ds}")
    return [r[col] for r in rows]


def union(ds: str, mans: dict, names, dim: str) -> set:
    return {v for n in names for v in values(ds, mans[n], dim)}


def names_of(v, names) -> list:
    return list(names) if v == "*" else as_list(v)


def rule_problems(ds: str, doc: dict) -> list:
    """Struttura del file delle regole: dataset, manifest dichiarati (file, righe e SHA-256 attesi), regole con id univoco,
    tipo, severità, dimensioni del dataset e manifest noti; campi specifici per tipo; nessun campo sconosciuto."""
    problems = []
    if doc.get("dataset") != ds:
        problems.append(f"'dataset' must be {ds!r}")
    mans = doc.get("manifests")
    if not isinstance(mans, dict) or not mans:
        return problems + ["'manifests' must be a non-empty object {name: {file, expected_rows, expected_sha256}}"]
    names = list(mans)
    unknown = sorted(set(names) - set(SPEC[ds]["manifests"]))
    if unknown:
        problems.append(f"unknown manifests {unknown} (expected names: {sorted(SPEC[ds]['manifests'])})")
    for n, m in mans.items():
        if not isinstance(m, dict) or m.get("file") != SPEC[ds]["manifests"].get(n, (None,))[0]:
            problems.append(f"manifest {n!r}: 'file' must be {SPEC[ds]['manifests'].get(n, ('?',))[0]!r}"); continue
        if type(m.get("expected_rows")) is not int or m["expected_rows"] < 1:
            problems.append(f"manifest {n!r}: 'expected_rows' must be an integer >= 1")
        if not SHA_RE.match(str(m.get("expected_sha256") or "")):
            problems.append(f"manifest {n!r}: 'expected_sha256' must be a SHA-256 (64 lower-case hexadecimal characters)")
    rules = doc.get("rules")
    if not isinstance(rules, list) or not rules:
        return problems + ["'rules' must be a non-empty list"]
    ids = [r.get("id") for r in rules if isinstance(r, dict)]
    problems += [f"duplicated rule id {i!r}" for i, c in Counter(ids).items() if c > 1]
    dims = tuple(SPEC[ds]["dimensions"])
    allowed = {"disjoint": {"a", "b"}, "subset": {"a", "b"}, "intersection": {"a", "b", "expected"},
               "no_internal_duplicates": {"manifests"}, "internal_duplicates": {"manifests", "expected"},
               "partition": {"parts", "sources"}, "equals_spread": {"a", "source", "per_class"},
               "spread_partition": {"parts", "source", "per_class"}, "key_consistency": {"manifests"},
               "celeba_conversion": {"source", "training", "validation", "eval"}, "counts": {"expected"},
               "declared_not_checkable": {"reason"}}
    for r in rules:
        if not isinstance(r, dict):
            problems.append(f"rule {r!r} is not an object"); continue
        rid, kind = r.get("id"), r.get("kind")
        if not isinstance(rid, str) or not rid:
            problems.append(f"rule without an id: {r}")
        if kind not in KINDS:
            problems.append(f"{rid}: kind {kind!r} not in {KINDS}"); continue
        if r.get("severity") not in SEVERITIES:
            problems.append(f"{rid}: severity {r.get('severity')!r} not in {SEVERITIES}")
        if not isinstance(r.get("description"), str) or not r["description"]:
            problems.append(f"{rid}: 'description' must be a non-empty text")
        extra = set(r) - {"id", "kind", "severity", "description", "by"} - allowed[kind]
        if extra:
            problems.append(f"{rid}: fields {sorted(extra)} not allowed for kind {kind}")
        missing = allowed[kind] - set(r)
        if missing:
            problems.append(f"{rid}: missing fields {sorted(missing)}")
        if kind == "declared_not_checkable":
            if r.get("severity") != "report":
                problems.append(f"{rid}: a declared_not_checkable rule is report-only")
            if "by" in r:
                problems.append(f"{rid}: a declared_not_checkable rule has no 'by'")
            continue
        by = r.get("by")
        if not isinstance(by, list) or not by or any(d not in dims for d in by) or len(set(by)) != len(by):
            problems.append(f"{rid}: 'by' must be a non-empty list of distinct dimensions of {ds} {dims}")
        if kind in KEY_ONLY and by != ["key"]:
            problems.append(f"{rid}: {kind} rules compare by key ('by': ['key'])")
        for f in ("a", "b", "manifests", "parts", "sources"):
            if f in r:
                vals = names_of(r[f], names)
                if not vals or any(not isinstance(n, str) or n not in names for n in vals):
                    problems.append(f"{rid}: {f} = {r[f]!r} names unknown manifests (declared: {sorted(names)})")
                elif len(set(vals)) != len(vals):
                    problems.append(f"{rid}: {f} lists the same manifest twice")
        for f in ("source", "training", "validation", "eval"):
            if f in r and (not isinstance(r[f], str) or r[f] not in names):
                problems.append(f"{rid}: {f} = {r[f]!r} must name one declared manifest")
        if kind in ("disjoint", "subset", "intersection") and "a" in r and "b" in r:
            if set(names_of(r["a"], names)) & set(names_of(r["b"], names)):
                problems.append(f"{rid}: the same manifest is on both sides (a and b must differ)")
        if kind == "partition" and set(as_list(r.get("parts"))) & set(as_list(r.get("sources"))):
            problems.append(f"{rid}: a part cannot be one of its sources")
        if kind in ("equals_spread", "spread_partition"):
            if type(r.get("per_class")) is not int or r["per_class"] < 1:
                problems.append(f"{rid}: 'per_class' must be an integer >= 1")
            if r.get("source") in as_list(r.get("a", r.get("parts"))):
                problems.append(f"{rid}: the derived manifest cannot be its own source")
            if kind == "equals_spread" and isinstance(r.get("a"), list):
                problems.append(f"{rid}: 'a' must be a single manifest")
        if kind == "intersection":
            e = r.get("expected")
            if not ((type(e) is int and e >= 0) or (isinstance(e, dict) and set(e) == set(by or []) and all(type(v) is int and v >= 0 for v in e.values()))):
                problems.append(f"{rid}: 'expected' must be an integer >= 0, or an object with one integer >= 0 for each dimension of 'by'")
        if kind == "internal_duplicates":
            e = r.get("expected")
            if not isinstance(e, dict) or not e or any(n not in names_of(r.get("manifests"), names) or type(v) is not int or v < 0 for n, v in e.items()):
                problems.append(f"{rid}: 'expected' must map manifests of the rule to integers >= 0 (unlisted manifests expect 0)")
        if kind == "counts":
            e = r.get("expected")
            if not isinstance(e, dict) or set(e) != set(names) or any(
                    not isinstance(v, dict) or set(v) != set(count_fields(ds)) or any(type(x) is not int or x < 0 for x in v.values())
                    for v in e.values()):
                problems.append(f"{rid}: 'expected' must give, for every declared manifest, the integers {list(count_fields(ds))}")
        if kind == "celeba_conversion" and ds != "celeba_spoof":
            problems.append(f"{rid}: celeba_conversion applies to celeba_spoof only")
    return problems


def outcome(ok: bool, must: bool) -> str:
    return "pass" if ok else ("FAIL" if must else "reported")


def spread_keys(rows, per_class: int) -> list:
    """Chiavi di `spread(chiavi ordinate, per_class)` per classe, bona fide e poi attacchi: lo stesso campionamento di
    build_arrays (cartella ordinata) quando le chiavi hanno un prefisso comune (CASIA: lo split; CelebA: celeba_)."""
    out = []
    for lab in ("0", "1"):
        out += spread([r["key"] for r in rows if r["label"] == lab], per_class)
    return out


def celeba_conversion_sets(parquet_rows) -> dict:
    """Regola che riproduce le cartelle preparate di CelebA-Spoof (la procedura originale non è versionata): per classe,
    sulle righe con immagine in ordine di riga, eval = spread(righe, 150); training = le prime 1.500 righe restanti;
    validation = eval più le 300 righe successive; le altre restano inutilizzate."""
    out = {"training": [], "validation": [], "eval": []}
    for lab in ("0", "1"):
        rows = sorted((r for r in parquet_rows if r["label"] == lab), key=lambda r: int(r["row"]))
        keys = [r["key"] for r in rows]
        ev = set(spread(keys, EVAL_PER_CLASS))
        rest = [k for k in keys if k not in ev]
        out["eval"] += sorted(ev)
        out["training"] += rest[:CELEBA_N_TRAINING]
        out["validation"] += sorted(ev) + rest[CELEBA_N_TRAINING:CELEBA_N_TRAINING + CELEBA_N_VALIDATION_EXTRA]
    return out


def multiset_check(expected: Counter, actual: Counter, must: bool, base: dict) -> dict:
    missing, extra = expected - actual, actual - expected
    c = {**base, "n_expected": sum(expected.values()), "n_rows": sum(actual.values()), "n_missing": sum(missing.values()),
         "n_extra": sum(extra.values())}
    ok = not missing and not extra
    c["outcome"] = outcome(ok, must)
    if not ok and must:
        c["examples"] = examples_of(set(missing) | set(extra))
    return c


def evaluate_rules(ds: str, mans: dict, rules: list) -> list:
    """Verdetto di ogni regola. Per ogni verifica: conteggi, esito (pass, FAIL, reported) ed esempi solo per le verifiche
    "must" fallite. Le regole declared_not_checkable danno il verdetto "not checkable" con il motivo dichiarato."""
    verdicts = []
    names = list(mans)
    for r in rules:
        must, kind = r["severity"] == "must", r["kind"]
        checks, involved = [], []
        if kind == "declared_not_checkable":
            verdicts.append({"id": r["id"], "kind": kind, "severity": r["severity"], "by": [], "description": r["description"],
                             "verdict": NOT_CHECKABLE, "checks": [{"by": "-", "outcome": NOT_CHECKABLE, "reason": r["reason"]}]})
            continue
        if kind in ("disjoint", "subset", "intersection"):
            a, b = names_of(r["a"], names), names_of(r["b"], names)
            involved = a + b
            for dim in r["by"]:
                if kind == "disjoint":
                    for an in a:
                        for bn in b:
                            va, vb = values(ds, mans[an], dim), values(ds, mans[bn], dim)
                            shared = set(va) & set(vb)
                            c = {"a": an, "b": bn, "by": dim, "n_shared": len(shared), "rows_a": sum(v in shared for v in va),
                                 "rows_b": sum(v in shared for v in vb), "outcome": outcome(not shared, must)}
                            if shared and must:
                                c["examples"] = examples_of(shared)
                            checks.append(c)
                elif kind == "subset":
                    ub = union(ds, mans, b, dim)
                    for an in a:
                        miss = set(values(ds, mans[an], dim)) - ub
                        c = {"a": an, "b": " + ".join(b), "by": dim, "n_not_in_b": len(miss), "outcome": outcome(not miss, must)}
                        if miss and must:
                            c["examples"] = examples_of(miss)
                        checks.append(c)
                else:
                    shared = union(ds, mans, a, dim) & union(ds, mans, b, dim)
                    exp = r["expected"] if type(r["expected"]) is int else r["expected"][dim]
                    c = {"a": " + ".join(a), "b": " + ".join(b), "by": dim, "n_shared": len(shared), "expected": exp,
                         "rows_a": sum(v in shared for n in a for v in values(ds, mans[n], dim)),
                         "rows_b": sum(v in shared for n in b for v in values(ds, mans[n], dim)),
                         "outcome": outcome(len(shared) == exp, must)}
                    checks.append(c)
        elif kind in ("no_internal_duplicates", "internal_duplicates"):
            involved = names_of(r["manifests"], names)
            exp = r.get("expected") or {}
            for n in involved:
                for dim in r["by"]:
                    cnt = Counter(values(ds, mans[n], dim))
                    dup = {v: k for v, k in cnt.items() if k > 1}
                    want = exp.get(n, 0)
                    c = {"manifest": n, "by": dim, "duplicated_values": len(dup), "rows": sum(dup.values()),
                         "outcome": outcome(len(dup) == want, must)}
                    if kind == "internal_duplicates":
                        c["expected"] = want
                    if len(dup) != want and must and dup:
                        c["examples"] = examples_of(set(dup))
                    checks.append(c)
        elif kind == "partition":
            parts, sources = as_list(r["parts"]), as_list(r["sources"])
            involved = parts + sources
            expected = Counter(union(ds, mans, sources, "key"))
            actual = Counter(k for n in parts for k in values(ds, mans[n], "key"))
            checks.append(multiset_check(expected, actual, must, {"a": " + ".join(parts), "b": " + ".join(sources), "by": "key",
                                                                   "derivation": "partition"}))
        elif kind in ("equals_spread", "spread_partition"):
            parts = [r["a"]] if kind == "equals_spread" else as_list(r["parts"])
            involved = parts + [r["source"]]
            expected = Counter(spread_keys(mans[r["source"]], r["per_class"]))
            actual = Counter(k for n in parts for k in values(ds, mans[n], "key"))
            checks.append(multiset_check(expected, actual, must, {"a": " + ".join(parts), "b": r["source"], "by": "key",
                                                                   "derivation": f"{kind} ({r['per_class']} per class)"}))
        elif kind == "celeba_conversion":
            involved = [r["source"], r["training"], r["validation"], r["eval"]]
            sets = celeba_conversion_sets(mans[r["source"]])
            for part in ("training", "validation", "eval"):
                checks.append(multiset_check(Counter(sets[part]), Counter(values(ds, mans[r[part]], "key")), must,
                                             {"a": r[part], "b": r["source"], "by": "key", "derivation": f"conversion rule ({part})"}))
        elif kind == "counts":
            involved = list(r["expected"])
            for n, want in r["expected"].items():
                have = {k: counts(ds, mans[n], private=False)[k] for k in count_fields(ds)}
                for k in count_fields(ds):
                    checks.append({"manifest": n, "by": k, "expected": want[k], "count": have[k], "outcome": outcome(have[k] == want[k], must)})
        elif kind == "key_consistency":
            involved = names_of(r["manifests"], names)
            seen = defaultdict(dict)
            for n in involved:
                for row in mans[n]:
                    seen[row["key"]].setdefault(n, row)
            shared = {k: v for k, v in seen.items() if len(v) > 1}
            for field in SPEC[ds]["columns"][1:]:
                bad = sorted(k for k, by_man in shared.items() if len({x.get(field) for x in by_man.values()}) > 1)
                c = {"manifest": "all" if r["manifests"] == "*" else "+".join(involved), "by": f"key->{field}",
                     "keys_in_several_manifests": len(shared), "duplicated_values": len(bad), "rows": len(bad),
                     "outcome": outcome(not bad, must)}
                if bad and must:
                    c["examples"] = bad[:3]
                checks.append(c)
        # nessuna regola passa a vuoto: un manifest senza righe o una regola senza verifiche è un fallimento
        empty = sorted(n for n in set(involved) if not mans[n])
        if empty or not checks:
            checks.append({"by": "-", "outcome": "FAIL", "vacuous": True, "empty_manifests": empty})
        failed = any(c["outcome"] == "FAIL" for c in checks)
        if must or failed:
            verdict = "FAIL" if failed else "pass"
        else:
            verdict = "reported" if any(c["outcome"] == "reported" for c in checks) else "pass"
        verdicts.append({"id": r["id"], "kind": kind, "severity": r["severity"], "by": r["by"], "description": r["description"],
                         "verdict": verdict, "checks": checks})
    return verdicts


def overlap_matrix(ds: str, mans: dict) -> dict:
    names = list(mans)
    dims = [d for d in SPEC[ds]["dimensions"] if d != "local_subject"]
    pairs = {}
    for i, an in enumerate(names):
        for bn in names[i + 1:]:
            pairs[f"{an}|{bn}"] = {}
            for dim in dims:
                va, vb = values(ds, mans[an], dim), values(ds, mans[bn], dim)
                shared = set(va) & set(vb)
                pairs[f"{an}|{bn}"][dim] = {"n_shared": len(shared), "rows_a": sum(v in shared for v in va), "rows_b": sum(v in shared for v in vb)}
    distinct = {n: {dim: len(set(values(ds, mans[n], dim))) for dim in dims} | {"rows": len(mans[n])} for n in names}
    return {"manifests": names, "dimensions": dims, "distinct": distinct, "pairs": pairs}


def overlap_markdown(ds: str, rep: dict) -> str:
    names, dims = rep["matrix"]["manifests"], rep["matrix"]["dimensions"]
    L = [f"# {SPEC[ds]['title']} overlap report", "",
         f"Rules: `{rep['rules_file']}` (SHA-256 `{rep['rules_sha256']}`). Generated by `scripts/rgb_manifests.py check-overlap "
         f"--dataset {ds}`, commit `{rep['commit']}`{' (working tree modified)' if rep['dirty'] else ''}, {rep['checked']}.", "",
         f"**Result: {'PASS' if rep['passed'] else 'FAIL'}** ({plural(rep['n_must_failed'], 'must-hold rule')} failed, "
         f"{plural(rep['n_reported'], 'rule')} with reported overlaps, {plural(rep['n_not_checkable'], 'declaration')} not checkable on this copy).", "",
         "Duplicates are exact (file SHA-256 or identical key); near-duplicates (consecutive frames of a video, the same face in "
         "another capture) are not searched. This check does not establish absence of leakage.", "", "## Manifests", "",
         "| name | file | rows | " + " | ".join(f"distinct {d}" for d in dims) + " | SHA-256 |", "|---|---|---|" + "---|" * len(dims) + "---|"]
    for n in names:
        d, m = rep["matrix"]["distinct"][n], rep["manifests"][n]
        L.append(f"| {n} | `{m['file']}` | {d['rows']:,} | " + " | ".join(f"{d[x]:,}" for x in dims) + f" | `{m['sha256'][:16]}…` |")
    L += ["", "## Rule verdicts", "", "| rule | kind | severity | by | verdict | description |", "|---|---|---|---|---|---|"]
    for v in rep["rules"]:
        L.append(f"| {v['id']} | {v['kind']} | {v['severity']} | {', '.join(v['by']) or '-'} | **{v['verdict']}** | {v['description']} |")
    L += ["", "Checks with a non-zero count, every derivation and every declaration:", ""]
    for v in rep["rules"]:
        for c in v["checks"]:
            ex = f" (e.g. {', '.join(map(str, c['examples']))})" if c.get("examples") else ""
            if c.get("vacuous"):
                L.append(f"- {v['id']}: vacuous rule (empty manifests: {', '.join(c['empty_manifests']) or 'none'}) → FAIL")
            elif c["outcome"] == NOT_CHECKABLE:
                L.append(f"- {v['id']}: not checkable: {c['reason']}")
            elif "count" in c:
                if c["outcome"] != "pass":
                    L.append(f"- {v['id']} ({c['manifest']}, {c['by']}): {c['count']:,}, declared {c['expected']:,} → {c['outcome']}")
            elif "derivation" in c:
                L.append(f"- {v['id']} ({c['a']} from {c['b']}, {c['derivation']}): rows {c['n_rows']:,}, derived {c['n_expected']:,}, "
                         f"missing {c['n_missing']}, extra {c['n_extra']} → {c['outcome']}{ex}")
            elif "manifest" in c and (c["duplicated_values"] or "expected" in c or c["outcome"] != "pass"):
                exp = f", expected {c['expected']}" if "expected" in c else ""
                L.append(f"- {v['id']} ({c['manifest']}, by {c['by']}): duplicated values {c['duplicated_values']}, rows {c['rows']}{exp} → {c['outcome']}{ex}")
            elif "n_shared" in c and (c["n_shared"] or "expected" in c or c["outcome"] != "pass"):
                exp = f", expected {c['expected']}" if "expected" in c else ""
                L.append(f"- {v['id']} ({c['a']} vs {c['b']}, by {c['by']}): shared {c['n_shared']:,}, rows {c['rows_a']:,}/{c['rows_b']:,}{exp} → {c['outcome']}{ex}")
            elif "n_not_in_b" in c and (c["n_not_in_b"] or c["outcome"] != "pass"):
                L.append(f"- {v['id']} ({c['a']} in {c['b']}, by {c['by']}): not in b {c['n_not_in_b']} → {c['outcome']}{ex}")
    for dim in dims:
        L += ["", f"## Pairwise matrix: shared {dim} values (rows of the row manifest / rows of the column manifest)", "",
              "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for an in names:
            cells = []
            for bn in names:
                if an == bn:
                    cells.append(f"({rep['matrix']['distinct'][an][dim]:,})"); continue
                k = f"{an}|{bn}" if f"{an}|{bn}" in rep["matrix"]["pairs"] else f"{bn}|{an}"
                o = rep["matrix"]["pairs"][k][dim]
                ra, rb = (o["rows_a"], o["rows_b"]) if k.startswith(an + "|") else (o["rows_b"], o["rows_a"])
                cells.append(f"{o['n_shared']:,} ({ra:,}/{rb:,})" if o["n_shared"] else "0")
            L.append(f"| **{an}** | " + " | ".join(cells) + " |")
    L += ["", "Diagonal: number of distinct values in the manifest."]
    return "\n".join(L) + "\n"


# ============================================================================ identificativi per la scansione pubblica

CASIA_NAME_ANY = re.compile(r"\d+_(?:HR_)?\d\.avi_\d+_(?:real|fake)\.jpg")
SUBJECT_KEY_ANY = re.compile(r"(?<![A-Za-z0-9])(?:train|test):\d+(?::(?:HR_)?\d)?(?![0-9])")
CELEBA_KEY_ANY = re.compile(r"celeba_\d{5}")
HEX64 = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")


def record_problems(text: str, ids: dict) -> list:
    """Record per soggetto, per video o per immagine in un testo pubblico: chiavi di soggetto e di video CASIA, nomi di
    frame CASIA e chiavi CelebA dei manifest, SHA-256 di un'immagine dei manifest (ids: insiemi dei valori reali)."""
    bad = []
    found = {m.group(0) for m in SUBJECT_KEY_ANY.finditer(text)}
    found |= {":".join(f.split(":")[:2]) for f in found}           # anche la parte del soggetto di una chiave di video
    hits = found & (ids.get("subject_keys", set()) | ids.get("video_keys", set()))
    if hits:
        bad.append(f"{len(hits)} subject or video keys (e.g. {sorted(hits)[0]!r})")
    names = {m.group(0) for m in CASIA_NAME_ANY.finditer(text)} & ids.get("names", set())
    names |= {m.group(0) for m in CELEBA_KEY_ANY.finditer(text)} & ids.get("names", set())
    if names:
        bad.append(f"{len(names)} image names or keys (e.g. {sorted(names)[0]!r})")
    h = set(HEX64.findall(text)) & ids.get("hashes", set())
    if h:
        bad.append(f"{len(h)} per-image SHA-256 values (e.g. {sorted(h)[0][:16]}…)")
    return bad


def identifiers(ds: str, mans: dict) -> dict:
    rows = [r for rs in mans.values() for r in rs]
    if ds == "casia_fasd":
        return {"subject_keys": {r["subject_key"] for r in rows}, "video_keys": {r["video_key"] for r in rows},
                "names": {r["key"].split("/", 1)[1] for r in rows}, "hashes": {r["sha256"] for r in rows}}
    return {"names": {r["key"] for r in rows}, "hashes": {r["sha256"] for r in rows}}


def public_text_problems(text: str, ids: dict, markdown: bool = False) -> list:
    return public_problems(text, [], markdown=markdown) + record_problems(text, ids)


# ============================================================================ split delle CNN (logica del commit 0ac7143)

def official_split(y, groups):
    """train di 0ac7143: GroupShuffleSplit(test_size=0.2, random_state=42) se ci sono almeno 5 gruppi distinti,
    altrimenti train_test_split stratificato del 10 %. Restituisce (indici di fit, indici di validazione, ramo)."""
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    if groups is not None and len(set(groups.tolist())) >= 5:
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(np.zeros(len(y)), y, groups))
        return tr, va, "GroupShuffleSplit(test_size=0.2, random_state=42) on the s_train groups"
    tr, va = train_test_split(np.arange(len(y)), test_size=0.1, stratify=y, random_state=SEED)
    return tr, va, "train_test_split(test_size=0.1, stratify=y, random_state=42)"


def pooled_split(y_train, y_test):
    """pooled di 0ac7143: train e test della cache concatenati, train_test_split(train_size=0.8, stratify, 42), poi sul
    training (senza gruppi) validazione stratificata del 10 %. Indici sulla concatenazione."""
    from sklearn.model_selection import train_test_split
    yp = np.concatenate([y_train, y_test])
    tr, te = train_test_split(np.arange(len(yp)), train_size=0.8, stratify=yp, random_state=SEED)
    fi, va = train_test_split(np.arange(len(tr)), test_size=0.1, stratify=yp[tr], random_state=SEED)
    return tr[fi], tr[va], te


def sampling(ds: str, train_rows, test_rows):
    """Ordine di campionamento di build_arrays dai manifest: per classe (bona fide e poi attacchi) spread(chiavi
    ordinate, 1500); gruppi s_* come nella cache (CASIA: numero locale del soggetto; CelebA: nome del file, 0ac7143)."""
    by_key = {r["key"]: r for r in list(train_rows) + list(test_rows)}
    out = {}
    for split, rows in (("train", train_rows), ("test", test_rows)):
        keys = spread_keys(rows, PER_CLASS)
        y = np.array([int(by_key[k]["label"]) for k in keys], np.int64)
        s = np.array([by_key[k]["subject"] if ds == "casia_fasd" else f"{k}.jpg" for k in keys])
        out[split] = {"keys": keys, "y": y, "s": s}
    return out


def split_keys(ds: str, smp: dict) -> dict:
    tr, va, branch = official_split(smp["train"]["y"], smp["train"]["s"])
    keys_all = smp["train"]["keys"] + smp["test"]["keys"]
    pf, pv, pt = pooled_split(smp["train"]["y"], smp["test"]["y"])
    return {"cnn_train": [smp["train"]["keys"][i] for i in tr], "cnn_es": [smp["train"]["keys"][i] for i in va],
            "cnn_test": list(smp["test"]["keys"]), "pooled_train": [keys_all[i] for i in pf], "pooled_es": [keys_all[i] for i in pv],
            "pooled_test": [keys_all[i] for i in pt], "_branch": branch, "_idx": {"fit": tr, "val": va, "pooled_fit": pf, "pooled_val": pv, "pooled_test": pt}}


def held_out_uniqueness(rows, n_val: int, k: int, val_rows=None) -> dict:
    """Indipendente da scikit-learn: fra le combinazioni di k soggetti di training, quante danno esattamente n_val
    immagini di validazione (il conteggio delle schede) e quante anche gli stessi conteggi per classe della ricostruzione.
    Se il numero non è 1, la ricostruzione dipende dal generatore casuale di GroupShuffleSplit (dichiarato)."""
    per = Counter(r["subject"] for r in rows)
    per_cls = Counter((r["subject"], r["label"]) for r in rows)
    combos = [c for c in combinations(sorted(per, key=int), k) if sum(per[s] for s in c) == n_val]
    out = {"n_training_subjects": len(per), "k_held_out": k, "n_combinations": sum(1 for _ in combinations(per, k)),
           "n_with_card_n_val": len(combos)}
    if val_rows is not None:
        want = Counter(r["label"] for r in val_rows)
        same = [c for c in combos if all(sum(per_cls[(s, lab)] for s in c) == want[lab] for lab in ("0", "1"))]
        out["n_with_card_n_val_and_class_counts"] = len(same)
        out["reconstructed_is_among_them"] = sorted({r["subject"] for r in val_rows}, key=int) in [list(c) for c in same]
    out["note"] = ("the card records the total of the validation images only: the held-out subjects are not determined by the "
                   "counts alone, and the reconstruction relies on GroupShuffleSplit with seed 42 (scikit-learn " + BUILT_WITH["scikit-learn"]
                   + "); `reconstruct` reports any drift of the library explicitly")
    return out


def library_versions() -> dict:
    import sklearn
    return {"numpy": np.__version__, "scikit-learn": sklearn.__version__}


# ============================================================================ build

def checkpoint_linkage(ds: str, man_counts: dict, check) -> dict:
    """Schede official e pooled del dataset: SHA-256 del .h5 (scheda, models/weights/manifest.json, file), conteggi
    n_train/n_val/n_test e classi del test uguali ai manifest, dataset e protocollo della scheda."""
    wm_p = WEIGHTS / "manifest.json"
    wm = load_json(wm_p) if wm_p.is_file() else {}
    out = {}
    for protocol, archs, prefix, suffix in (("official", ARCHS, "cnn", ""), ("pooled", POOLED_ARCHS, "pooled", "-pooled")):
        exp = {"n_train": man_counts[f"{prefix}_train"]["n"], "n_val": man_counts[f"{prefix}_es"]["n"], "n_test": man_counts[f"{prefix}_test"]["n"]}
        for arch in archs:
            name = f"{arch}__{ds}{suffix}"
            card_p, h5 = WEIGHTS / f"{name}.json", WEIGHTS / f"{name}.h5"
            if not card_p.is_file():
                check(f"card_present:{name}", False, f"missing {relpath(card_p)}"); continue
            card = load_json(card_p)
            h5_sha = sha256(h5) if h5.is_file() else None
            test = card.get("test") or {}
            e = {"card": card_p.name, "weights_file": h5.name, "weights_sha256": card.get("weights_sha256"),
                 "weights_sha256_verified": h5_sha is not None and h5_sha == card.get("weights_sha256"),
                 "weights_manifest_sha256_equal": (wm.get(name) or {}).get("weights_sha256") == card.get("weights_sha256"),
                 "protocol": card.get("protocol"), "dataset": card.get("dataset"),
                 "val_split": VAL_SPLIT_EN.get(card.get("val_split"), card.get("val_split")),
                 "val_split_note": "English translation of the value recorded in the card", "saved": card.get("saved"),
                 "epochs_run": card.get("epochs_run"), "card_counts": {k: card.get(k) for k in exp},
                 "card_test_by_class": {"n_bona_fide": test.get("n_real"), "n_attack": test.get("n_attack")},
                 "manifests": {"n_train": f"{prefix}_train", "n_val": f"{prefix}_es", "n_test": f"{prefix}_test"}}
            e["counts_match_manifests"] = (e["card_counts"] == exp and test.get("n_real") == man_counts[f"{prefix}_test"]["n_bona_fide"]
                                           and test.get("n_attack") == man_counts[f"{prefix}_test"]["n_attack"])
            check(f"weights_sha256:{name}", e["weights_sha256_verified"], f"{h5.name} missing or different from the card")
            check(f"weights_manifest:{name}", e["weights_manifest_sha256_equal"], "models/weights/manifest.json differs from the card")
            check(f"card_counts:{name}", e["counts_match_manifests"], f"card {e['card_counts']} vs manifests {exp}")
            check(f"card_protocol:{name}", card.get("protocol") == protocol and card.get("dataset") == ds, "dataset or protocol of the card")
            out[name] = e
    return out


def derived_frames(npz: Path, smp_idx: dict, n_train: int) -> dict:
    """Frame derivati della cache (X_train, X_test) hashati come salvati: duplicati interni, fra train e test, fra
    fit e validazione dell'official, fra il lato di training dei pooled (fit + validazione) e il loro test."""
    z = np.load(npz)
    h = {s: frame_hashes(z[f"X_{s}"]) for s in ("train", "test")}
    shapes = {s: list(z[f"X_{s}"].shape) for s in ("train", "test")}
    out = {"hash": "SHA-256 of each preprocessed frame as stored in the cache (dtype, shape, C-order bytes)"}
    for s in ("train", "test"):
        c = Counter(h[s])
        dup = {v: n for v, n in c.items() if n > 1}
        out[s] = {"n_frames": len(h[s]), "shape": shapes[s], "distinct": len(c), "duplicated_values": len(dup), "duplicated_rows": sum(dup.values())}
    out["train_test_shared"] = len(set(h["train"]) & set(h["test"]))
    out["fit_vs_validation_shared"] = len({h["train"][i] for i in smp_idx["fit"]} & {h["train"][i] for i in smp_idx["val"]})
    allh = h["train"] + h["test"]
    side = {allh[i] for i in smp_idx["pooled_fit"]} | {allh[i] for i in smp_idx["pooled_val"]}
    out["pooled_trainval_vs_test_shared"] = len(side & {allh[i] for i in smp_idx["pooled_test"]})
    out["linkage"] = ("frames are linked to the image files by sampling order and the label and group arrays (y_*, s_*), "
                      "not by per-image path or hash: the cache stores no file names; frame bytes are never exported")
    out["_hashes"] = allh
    return out


def cache_check(npz: Path, smp: dict, check) -> dict:
    info = {"file": rel(npz), "present": npz.is_file()}
    if not npz.is_file():
        check("training_cache_present", False, f"missing {relpath(npz)}")
        return info
    z = np.load(npz)
    info["members"] = sorted(z.files)
    info["sha256"] = sha256(npz)
    for s in ("train", "test"):
        info[f"matches_{s}"] = bool(np.array_equal(z[f"y_{s}"], smp[s]["y"]) and np.array_equal(z[f"s_{s}"], smp[s]["s"]))
        info[f"n_{s}"] = int(len(z[f"y_{s}"]))
        check(f"training_cache_{s}", info[f"matches_{s}"], f"y_{s}/s_{s} of the cache differ from the sampling rebuilt from the folders")
    info["linkage"] = "by label and group arrays in sampling order (counts and order), not by per-image path or hash"
    return info


def casia_build(check):
    import tarfile
    src = {"dataset": "CASIA-FASD (Zhang et al., ICB 2012)", "mirror": f"Hugging Face dataset {CASIA_MIRROR}",
           "archive": CASIA_ARCHIVE.name, "readme_in_mirror": False, "split_lists_in_mirror": False, "videos_in_mirror": False}
    meta = hf_meta(CASIA_HF_META)
    src["revision"] = (meta or {}).get("revision")
    src["archive_sha256"] = sha256(CASIA_ARCHIVE) if CASIA_ARCHIVE.is_file() else None
    src["archive_bytes"] = CASIA_ARCHIVE.stat().st_size if CASIA_ARCHIVE.is_file() else None
    src["archive_sha256_equals_hf_etag"] = bool(meta and src["archive_sha256"] and meta["etag"] == src["archive_sha256"])
    check("archive_present", src["archive_sha256"] is not None, f"missing {relpath(CASIA_ARCHIVE)}")
    check("archive_matches_hf_metadata", src["archive_sha256_equals_hf_etag"], "archive SHA-256 differs from the etag of the cache metadata")
    if CASIA_ARCHIVE.is_file():
        per = Counter()
        with tarfile.open(CASIA_ARCHIVE) as tf:          # solo elenco, nessuna estrazione
            for m in tf:
                if m.isfile():
                    parts = m.name.split("/")
                    per[f"{parts[1]}/{parts[3]}" if len(parts) > 3 else m.name] += 1
        src["archive_files_per_folder"] = dict(sorted(per.items()))
    rows, depth = {"train": [], "test": []}, {}
    for split, sub in CASIA_SPLIT_DIR.items():
        d = CASIA_RAW / sub / "color"
        if not d.is_dir():
            fail(f"missing folder {relpath(d)}")
        names = sorted(p.name for p in d.iterdir() if p.is_file())
        dd = CASIA_RAW / sub / "depth"
        depth[split] = {"n": sum(1 for _ in dd.iterdir()) if dd.is_dir() else 0,
                        "same_names_as_color": dd.is_dir() and sorted(p.name for p in dd.iterdir()) == names, "used": False}
        bad = []
        for n in names:
            try:
                rows[split].append(casia_row(split, n, sha256(d / n), (d / n).stat().st_size))
            except ValueError as e:
                bad.append(str(e))
        check(f"mirror_names_valid:{split}", not bad, "; ".join(bad[:3]))
        units = Counter((r["subject"], r["video"]) for r in rows[split])
        subj = {r["subject"] for r in rows[split]}
        check(f"mirror_subjects:{split}", {int(s) for s in subj} == set(CASIA_SUBJECTS[split]), f"subjects {sorted(subj, key=int)}")
        check(f"mirror_videos_complete:{split}", all((s, v) in units for s in subj for v in CASIA_VIDEOS), "a (subject, video) is missing")
    src["depth_maps"] = depth
    by_key = {r["key"]: r for rs in rows.values() for r in rs}
    # cartelle preparate: stessi nomi del mirror per classe, stesso file (inode) o stesso SHA-256
    prepared, link = {}, Counter()
    for split, tag in CASIA_TAG.items():
        prepared[split] = []
        for cls, suf in (("bonafide", "real"), ("attack", "fake")):
            d = CASIA_IMAGES / f"{cls}_{tag}"
            if not d.is_dir():
                fail(f"missing folder {relpath(d)}")
            names = sorted(p.name for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
            expected = sorted(r["key"].split("/", 1)[1] for r in rows[split] if r["key"].endswith(f"_{suf}.jpg"))
            check(f"prepared_equals_mirror:{cls}_{tag}", names == expected, f"{d.name}: names differ from the mirror colour frames")
            for n in names:
                p, raw = d / n, CASIA_RAW / CASIA_SPLIT_DIR[split] / "color" / n
                if raw.is_file() and nm.same_file(p, raw):
                    link["hard_linked"] += 1
                    prepared[split].append(dict(by_key[f"{split}/{n}"]))
                else:
                    link["copied"] += 1
                    prepared[split].append(casia_row(split, n, sha256(p), p.stat().st_size))
    # valutazione dell'app (congelata): file del test del mirror, solo confronto
    ev = []
    for cls in ("real", "attack"):
        d = CASIA_EVAL / cls
        if not d.is_dir():
            fail(f"missing folder {relpath(d)}")
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                r = casia_row("test", p.name, sha256(p), p.stat().st_size)
                check(f"eval_class:{cls}", (cls == "real") == (r["label"] == "0"), f"{p.name} in {cls}")
                ev.append(r)
    src["prepared_vs_mirror"] = dict(link)
    base = {"mirror_train": rows["train"], "mirror_test": rows["test"], "prepared_training": prepared["train"],
            "prepared_validation": prepared["test"], "eval": ev}
    smp = sampling("casia_fasd", prepared["train"], prepared["test"])
    return src, base, smp


def hf_meta(p: Path):
    if not p.is_file():
        return None
    lines = p.read_text(encoding="utf-8").splitlines()
    out = {"revision": lines[0].strip() if lines else None, "etag": lines[1].strip() if len(lines) > 1 else None}
    if len(lines) > 2:
        try:
            out["downloaded"] = time.strftime("%Y-%m-%d", time.gmtime(float(lines[2])))
        except ValueError:
            pass
    return out


def celeba_build(check):
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        fail("missing Python package 'pyarrow': needed to read the CelebA-Spoof parquet (build only)")
    src = {"dataset": "CelebA-Spoof (Zhang et al., ECCV 2020)", "mirror": f"Hugging Face dataset {CELEBA_MIRROR}",
           "file": CELEBA_SHARD, "shard": "shard 0 of 10 of the test split", "readme_in_mirror": False}
    meta = hf_meta(CELEBA_HF_META)
    src["revision"] = (meta or {}).get("revision")
    if not CELEBA_PARQUET.is_file():
        fail(f"missing {relpath(CELEBA_PARQUET)}")
    src["file_sha256"] = sha256(CELEBA_PARQUET)
    src["file_bytes"] = CELEBA_PARQUET.stat().st_size
    src["file_sha256_equals_hf_etag"] = bool(meta and meta["etag"] == src["file_sha256"])
    check("parquet_matches_hf_metadata", src["file_sha256_equals_hf_etag"], "parquet SHA-256 differs from the etag of the cache metadata")
    pf = pq.ParquetFile(CELEBA_PARQUET)
    t = pf.read()
    src["columns"] = t.schema.names
    check("parquet_columns", t.schema.names == ["cropped_image", "labels", "labelNames"], f"columns {t.schema.names}")
    img = t.column("cropped_image").combine_chunks()
    data, paths = img.field("bytes").to_pylist(), img.field("path").to_pylist()
    labels, lnames = t.column("labels").to_pylist(), t.column("labelNames").to_pylist()
    src["rows"], src["row_groups"] = len(labels), pf.metadata.num_row_groups
    check("parquet_rows", len(labels) == CELEBA_ROWS, f"{len(labels)} rows, expected {CELEBA_ROWS}")
    check("parquet_labels", all(l in (0, 1) and n == CELEBA_LABEL_NAME[str(l)] for l, n in zip(labels, lnames)), "labels/labelNames")
    src["path_empty_rows"] = sum(p in (None, "") for p in paths)
    null = [i for i, b in enumerate(data) if b is None]
    src["rows_without_image"] = len(null)
    src["rows_without_image_by_label"] = {"bona_fide" if k == 0 else "attack": v for k, v in sorted(Counter(labels[i] for i in null).items())}
    src["png_rows"] = sum(b[:8] == b"\x89PNG\r\n\x1a\n" for b in data if b is not None)
    check("parquet_png", src["png_rows"] == len(labels) - len(null), "not every image is a PNG")
    parquet = [celeba_row(i, labels[i], hashlib.sha256(b).hexdigest(), len(b)) for i, b in enumerate(data) if b is not None]
    by_key = {r["key"]: r for r in parquet}
    del data, t, img
    prepared, ident = {}, Counter()
    for tag in ("training", "validation"):
        prepared[tag] = []
        for cls, lab in (("bonafide", "0"), ("attack", "1")):
            d = CELEBA_IMAGES / f"{cls}_{tag}"
            if not d.is_dir():
                fail(f"missing folder {relpath(d)}")
            for p in sorted(d.iterdir()):
                if p.suffix.lower() not in IMG_EXT:
                    continue
                k = p.stem
                pr = by_key.get(k)
                h = sha256(p)
                ok = pr is not None and pr["sha256"] == h and pr["label"] == lab
                ident["byte_identical_and_same_label" if ok else "different"] += 1
                prepared[tag].append(dict(pr) if ok else {"key": k, "label": lab, "label_name": CELEBA_LABEL_NAME[lab],
                                                           "row": k[7:].lstrip("0") or "0", "official_split": "test", "sha256": h, "bytes": str(p.stat().st_size)})
    check("prepared_byte_identical_to_parquet", ident["different"] == 0, f"{ident['different']} prepared files differ from their parquet row")
    ev = []
    for cls, lab in (("real", "0"), ("attack", "1")):
        d = CELEBA_EVAL / cls
        if not d.is_dir():
            fail(f"missing folder {relpath(d)}")
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                pr = by_key.get(p.stem)
                h = sha256(p)
                check(f"eval_file:{p.stem[:7]}", pr is not None and pr["sha256"] == h and pr["label"] == lab, "evaluation file differs from its parquet row")
                ev.append(dict(pr) if pr else {"key": p.stem, "label": lab, "label_name": CELEBA_LABEL_NAME[lab], "row": "0",
                                               "official_split": "test", "sha256": h, "bytes": str(p.stat().st_size)})
    src["prepared_vs_parquet"] = dict(ident)
    base = {"parquet": parquet, "prepared_training": prepared["training"], "prepared_validation": prepared["validation"], "eval": ev}
    smp = sampling("celeba_spoof", prepared["training"], prepared["validation"])
    private = {"rows_without_image": null, "exact_duplicate_rows": sorted(sorted(int(by_key[k]["row"]) for k in g)
                                                                          for g in dup_groups(parquet))}
    return src, base, smp, private


def dup_groups(rows) -> list:
    g = defaultdict(list)
    for r in rows:
        g[r["sha256"]].append(r["key"])
    return [v for v in g.values() if len(v) > 1]


def cmd_build(a) -> None:
    ds = a.dataset
    out = Path(a.out or ds_out(Path(a.out_root), ds))
    for forbidden in (DATA, WEIGHTS, RESULTS / "c1", RESULTS / "eval", RESULTS / "m1", RESULTS / "m2"):
        if is_under(out, forbidden):
            fail(f"output folder {relpath(out)} is inside {relpath(forbidden)}: refused (read-only area)")
    checks, problems = {}, []

    def check(name, ok, msg=None):
        checks[name] = bool(ok)
        if not ok:
            problems.append(f"{name}: {msg}" if msg else name)
        return ok

    print(f"== {ds}: source, mirror and prepared folders", flush=True)
    private_extra = {}
    if ds == "casia_fasd":
        src, base, smp = casia_build(check)
        cache_p = CASIA_CACHE
    else:
        src, base, smp, private_extra = celeba_build(check)
        cache_p = CELEBA_CACHE
    print(f"   {', '.join(f'{n} {len(r):,}' for n, r in base.items())}", flush=True)

    print("== training cache and splits of the checkpoints (logic of commit 0ac7143)", flush=True)
    cache = cache_check(cache_p, smp, check)
    sp = split_keys(ds, smp)
    rows_by_key = {r["key"]: r for rs in base.values() for r in rs}
    sets = dict(base)
    for n in ("cnn_train", "cnn_es", "cnn_test", "pooled_train", "pooled_es", "pooled_test"):
        sets[n] = [rows_by_key[k] for k in sp[n]]
    reconstruction = {"logic": f"scripts/train_cnn.py at commit {SPLIT_LOGIC_COMMIT} (first versioned version; the training run of "
                               "12 September 2026 used an earlier, unversioned version)",
                      "sampling": f"spread(sorted file list, {PER_CLASS}) per class, bona fide first, training and test folders",
                      "official": sp["_branch"] + (" (groups = local subject number)" if ds == "casia_fasd" else
                                                   " (groups = file names, one image per group: hypothesis 1 of the inventory)"),
                      "pooled": "train_test_split(train_size=0.8, stratify, random_state=42) on train + test of the cache, then "
                                "train_test_split(test_size=0.1, stratify, random_state=42) on the training part (no groups)",
                      "library_versions": library_versions(), "built_with": BUILT_WITH, "training_cache": cache,
                      "original_file_list_recorded": False}
    if ds == "casia_fasd":
        tr_rows = [rows_by_key[k] for k in smp["train"]["keys"]]
        u = held_out_uniqueness(tr_rows, len(sp["cnn_es"]), 4, [rows_by_key[k] for k in sp["cnn_es"]])
        check("held_out_combination_consistent", u["reconstructed_is_among_them"], "the reconstructed held-out subjects do not give the counts")
        reconstruction["held_out_uniqueness"] = u
    else:
        from sklearn.model_selection import train_test_split
        y = smp["train"]["y"]
        _, va2 = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
        reconstruction["alternative_hypothesis"] = {
            "split": "train_test_split(test_size=0.2, stratify=y, random_state=42) (hypothesis 2 of the inventory)",
            "n_train": int(len(y) - len(va2)), "n_val": int(len(va2)),
            "val_by_class": {"bona_fide": int((y[va2] == 0).sum()), "attack": int((y[va2] == 1).sum())},
            "note": "gives the same totals as the cards (the cards record no per-class counts of the early stopping); which logic the training run used is not verified"}
    derived = None
    if cache_p.is_file():
        derived = derived_frames(cache_p, sp["_idx"], len(smp["train"]["keys"]))
        cache["derived_frames"] = {k: v for k, v in derived.items() if not k.startswith("_")}
        print(f"   derived frames: train {derived['train']['distinct']:,}/{derived['train']['n_frames']:,} distinct, test "
              f"{derived['test']['distinct']:,}/{derived['test']['n_frames']:,} distinct, train/test shared {derived['train_test_shared']}, "
              f"fit/validation shared {derived['fit_vs_validation_shared']}, pooled training side/test shared {derived['pooled_trainval_vs_test_shared']}", flush=True)

    print("== manifests", flush=True)
    man_dir = out / "manifests"
    manifests, manifests_public, man_counts = {}, {}, {}
    for name, (fname, desc) in SPEC[ds]["manifests"].items():
        h = write_manifest(ds, man_dir / fname, sets[name])
        base_info = {"file": f"manifests/{fname}", "description": desc, "sha256": h}
        man_counts[name] = counts(ds, sets[name], private=False)
        manifests[name] = {**base_info, **counts(ds, sets[name], private=True)}
        manifests_public[name] = {**base_info, **man_counts[name]}
        print(f"   {fname}: {len(sets[name]):,} rows, sha256 {h}", flush=True)
    vp = {n: row_problems(ds, sets[n]) for n in sets}
    check("rows_valid", not any(vp.values()), f"{ {n: v for n, v in vp.items() if v} }")
    rp = rules_path(ds)
    declared = (load_json(rp).get("manifests") or {}) if rp.is_file() else {}
    differ = sorted(n for n, m in manifests.items() if not isinstance(declared.get(n), dict)
                    or declared[n].get("expected_sha256") != m["sha256"] or declared[n].get("expected_rows") != m["n"])
    drift = ""
    if differ and library_versions() != BUILT_WITH:
        drift = (f"; the library versions differ from those that wrote the declared manifests ({library_versions()} vs {BUILT_WITH}): "
                 "a different random split of scikit-learn may be a drift of the library rather than of the data (see `reconstruct`)")
    check("manifests_equal_declared_in_rules", rp.is_file() and not differ,
          f"manifests differ from {rel(rp)} (rows or SHA-256): {differ}{drift}; update the declared values only after review")

    print("== checkpoint linkage", flush=True)
    ckpt = checkpoint_linkage(ds, man_counts, check)
    for n, e in ckpt.items():
        print(f"   {n}: sha256 verified {e['weights_sha256_verified']}, manifest.json {e['weights_manifest_sha256_equal']}, "
              f"counts match {e['counts_match_manifests']}", flush=True)

    passed = not problems
    ids = identifiers(ds, sets)
    common = {"schema": f"m2-rgb/{ds}/1", "dataset": ds, "source": src,
              "source_declarations": {k: v for k, v in SOURCE_DECLARATIONS[ds]},
              "reconstruction": reconstruction,
              "checkpoints": ckpt, "limits": list(LIMITS[ds]), "supported_comparisons": list(SUPPORTED[ds]),
              "checks": checks, "passed": passed, "problems": problems, **repo_state(), "created": now()}
    private = {"early_stopping_subjects": sorted({r["subject_key"] for r in sets["cnn_es"]})} if ds == "casia_fasd" else {}
    private.update(private_extra)
    if derived:
        dupk = [sorted(k for i, k in enumerate(smp["train"]["keys"] + smp["test"]["keys"]) if derived["_hashes"][i] == v)
                for v, c in Counter(derived["_hashes"]).items() if c > 1]
        private["derived_frame_duplicates"] = dupk
    summary = {**common, "manifests": manifests, "private": True, "restricted_records": private}
    aggregate = {**common, "manifests": manifests_public, "recount": recount_block(ds, sets), "private": False,
                 "note": "aggregate counts only: per-image manifests, subject and video keys and row indices are restricted"}
    out.mkdir(parents=True, exist_ok=True)
    summ_p, agg_p = out / f"{ds}_manifests_summary.json", out / f"{ds}_manifests_aggregate.json"
    summ_p.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    agg_text = json.dumps(aggregate, indent=1) + "\n"
    bad = public_text_problems(agg_text, ids)
    if bad:
        problems.append("public aggregate contains forbidden content: " + "; ".join(bad))
        aggregate.update(passed=False, problems=problems)
        agg_text = json.dumps(aggregate, indent=1) + "\n"
    agg_p.write_text(agg_text, encoding="utf-8")
    print(f"== {relpath(summ_p)} (private)\n== {relpath(agg_p)} (public aggregate, scan: {'clean' if not bad else 'FORBIDDEN CONTENT'})", flush=True)
    if problems:
        fail("build checks failed: " + "; ".join(problems))
    print(f"== build checks passed ({len(checks)} checks)", flush=True)


# ============================================================================ check-overlap e self-test

def load_declared(ds: str, rules_p: Path, man_dir: Path):
    """Regole e manifest dichiarati: struttura delle regole, righe e SHA-256 attesi (prima di ogni regola), validazione
    delle righe. Restituisce (doc, manifest, info) oppure solleva ManifestError per righe non valide."""
    doc = load_json(rules_p)
    problems = rule_problems(ds, doc)
    if problems:
        fail(f"{relpath(rules_p)}: " + "; ".join(problems))
    info, mismatch = {}, []
    for n, m in doc["manifests"].items():
        p = man_dir / m["file"]
        if not p.is_file():
            fail(f"missing {relpath(p)}")
        h = sha256(p)
        n_rows = len(read_csv(p, SPEC[ds]["columns"]))
        info[n] = {"file": m["file"], "sha256": h, "rows": n_rows}
        if n_rows != m["expected_rows"]:
            mismatch.append(f"{n}: {n_rows} rows, expected {m['expected_rows']}")
        if h != m["expected_sha256"]:
            mismatch.append(f"{n}: SHA-256 {h[:16]}… differs from the declared {m['expected_sha256'][:16]}…")
    if mismatch:
        fail(f"manifests differ from those declared in {relpath(rules_p)}: " + "; ".join(mismatch))
    mans = {n: load_manifest(ds, man_dir / m["file"]) for n, m in doc["manifests"].items()}
    return doc, mans, info


def cmd_check_overlap(a) -> None:
    ds = a.dataset
    rules_p = Path(a.rules or rules_path(ds))
    man_dir = Path(a.manifests_dir or ds_out(Path(a.out_root), ds) / "manifests")
    out = Path(a.out or ds_out(Path(a.out_root), ds))
    out.mkdir(parents=True, exist_ok=True)
    try:
        doc, mans, info = load_declared(ds, rules_p, man_dir)
    except ManifestError as e:
        # righe non valide: nessuna regola gira; il rapporto registra la validazione fallita (per il self-test)
        (out / "overlap_report.json").write_text(json.dumps({"dataset": ds, "passed": False, "row_validation_failed": True,
                                                             "rules": [], **repo_state(), "checked": now()}, indent=1) + "\n", encoding="utf-8")
        fail(str(e))
    verdicts = evaluate_rules(ds, mans, doc["rules"])
    n_fail = sum(v["verdict"] == "FAIL" for v in verdicts)
    rep = {"dataset": ds, "rules_file": rel(rules_p), "rules_sha256": sha256(rules_p), "manifests_dir": rel(man_dir), "manifests": info,
           "row_validation_failed": False, "rules": verdicts, "matrix": overlap_matrix(ds, mans), "n_must_failed": n_fail,
           "n_reported": sum(v["verdict"] == "reported" for v in verdicts),
           "n_not_checkable": sum(v["verdict"] == NOT_CHECKABLE for v in verdicts), "passed": n_fail == 0,
           "note": "restricted version: example keys are listed for failed must-hold checks", **repo_state(), "checked": now()}
    pub = json.loads(json.dumps(rep))
    for v in pub["rules"]:
        for c in v["checks"]:
            c.pop("examples", None)
    pub["note"] = "public version: counts and verdicts only, no keys, no subject or video identifiers"
    (out / "overlap_report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    (out / "overlap_report.md").write_text(overlap_markdown(ds, rep), encoding="utf-8")
    ids = identifiers(ds, mans)
    pub_json, pub_md = json.dumps(pub, indent=1) + "\n", overlap_markdown(ds, pub)
    bad = public_text_problems(pub_json, ids) + public_text_problems(pub_md, ids, markdown=True)
    if bad:
        for f in ("overlap_report_public.json", "overlap_report_public.md"):
            (out / f).unlink(missing_ok=True)
        fail("public overlap report contains forbidden content (not written): " + "; ".join(sorted(set(bad))))
    (out / "overlap_report_public.json").write_text(pub_json, encoding="utf-8")
    (out / "overlap_report_public.md").write_text(pub_md, encoding="utf-8")
    for v in verdicts:
        print(f"   {v['id']:<34} {v['severity']:<7} {v['verdict']}", flush=True)
    print(f"== {relpath(out / 'overlap_report.json')} and .md (restricted), overlap_report_public.json and .md (public)", flush=True)
    if n_fail:
        fail(f"{n_fail} must-hold overlap rules failed: " + ", ".join(v["id"] for v in verdicts if v["verdict"] == "FAIL"))
    print(f"== overlap checks passed ({plural(len(verdicts), 'rule')}, {rep['n_reported']} with reported overlaps, "
          f"{rep['n_not_checkable']} not checkable)", flush=True)
    if a.self_test:
        print("== self-test on faulty copies of the manifests", flush=True)
        if not run_self_test(ds, doc, rules_p, man_dir, mans, out):
            fail("self-test failed: a faulty copy was not rejected by exactly the expected rules (overlap_self_test.md)")
        print(f"== self-test passed: {relpath(out / 'overlap_self_test.json')} and .md", flush=True)


ROW_VALIDATION = "row-validation"


def self_test_faults(ds: str, m: dict) -> list:
    """Guasti del self-test: (id, descrizione, manifest modificati, regole che devono fallire ed esattamente quelle).
    Le righe scelte sono le prime in ordine di chiave che rispettano la condizione."""
    first = lambda rows, cond=lambda r: True: next(r for r in sorted(rows, key=lambda r: r["key"]) if cond(r))   # noqa: E731
    swap = lambda rows, x, y: [dict(r, sha256=y["sha256"]) if r is x else dict(r, sha256=x["sha256"]) if r is y else r for r in rows]  # noqa: E731
    trainval = {r["key"] for r in m["pooled_train"]} | {r["key"] for r in m["pooled_es"]}
    eval_keys = {r["key"] for r in m["eval"]}
    t0, t1 = sorted(m["cnn_test"], key=lambda r: r["key"])[:2]
    es0 = first(m["cnn_es"])
    if ds == "casia_fasd":
        P = "C"
        test_videos = Counter(r["video_key"] for r in m["pooled_test"])
        drop = first(m["pooled_test"], lambda r: test_videos[r["video_key"]] > 1)
        ev_old = first(m["eval"], lambda r: r["label"] == "1")
        cnn_test_keys = {r["key"] for r in m["cnn_test"]}
        ev_new = first(m["mirror_test"], lambda r: r["label"] == "1" and r["key"] not in eval_keys and r["key"] in cnn_test_keys
                       and (r["key"] in trainval) == (ev_old["key"] in trainval))
        mt = first(m["mirror_test"])
        pv = first(m["prepared_validation"])
        mv = first(m["cnn_train"])["subject_key"]
        faults = [
            ("F1-es-row-in-train", "first early-stopping row added to the CNN train", {"cnn_train": m["cnn_train"] + [es0]},
             {f"{P}07-cnn-split-partition", f"{P}08-fit-vs-es", f"{P}18-declared-counts"}),
            ("F2-sha-swapped-in-cnn-test", "SHA-256 of the first two CNN test rows swapped (mirror and prepared manifests unchanged)",
             {"cnn_test": swap(m["cnn_test"], t0, t1)}, {f"{P}17-key-consistency"}),
            ("F3-mirror-split-out-of-domain", "mirror_split 'dev' in the first CNN test row (outside train/test)",
             {"cnn_test": [dict(r, mirror_split="dev") if r is t0 else r for r in m["cnn_test"]]}, {ROW_VALIDATION}),
            ("F4-label-inconsistent-with-key", "label 1 in the first bona fide early-stopping row",
             {"cnn_es": [dict(r, label="1") if r is first(m["cnn_es"], lambda x: x["label"] == "0") else r for r in m["cnn_es"]]}, {ROW_VALIDATION}),
            ("F5-duplicated-pooled-test-row", "first pooled test row duplicated", {"pooled_test": m["pooled_test"] + [m["pooled_test"][0]]},
             {f"{P}01-internal-duplicates", f"{P}13-pooled-partition", f"{P}18-declared-counts"}),
            ("F6-test-hash-under-train-key", "the first CNN train row gets the SHA-256 of the first test-side frame",
             {"cnn_train": [dict(r, sha256=mt["sha256"]) if r is first(m["cnn_train"]) else r for r in m["cnn_train"]]},
             {f"{P}09-train-side-vs-test-side", f"{P}17-key-consistency"}),
            ("F7-pooled-test-row-removed", "a pooled test row removed (its video keeps other test frames)",
             {"pooled_test": [r for r in m["pooled_test"] if r is not drop]}, {f"{P}13-pooled-partition", f"{P}18-declared-counts"}),
            ("F8-eval-row-replaced", "an evaluation attack replaced by another test attack of the CNN test (same pooled side)",
             {"eval": [ev_new if r is ev_old else r for r in m["eval"]]}, {f"{P}11-eval-spread"}),
            ("F9-prepared-row-removed", "first row of the prepared test folders removed",
             {"prepared_validation": [r for r in m["prepared_validation"] if r is not pv]}, {f"{P}06-prepared-validation-equals-mirror", f"{P}18-declared-counts"}),
            ("F10-subject-moved-fit-to-es", "every frame of one fitting subject moved to the early stopping (partition and disjointness still hold)",
             {"cnn_train": [r for r in m["cnn_train"] if r["subject_key"] != mv], "cnn_es": m["cnn_es"] + [r for r in m["cnn_train"] if r["subject_key"] == mv]},
             {f"{P}18-declared-counts"}),
        ]
    else:
        P = "S"
        dup_keys = {k for g in dup_groups(m["parquet"]) for k in g}
        pt = first(m["prepared_training"], lambda r: r["key"] not in dup_keys)
        ev_old = first(m["eval"], lambda r: r["label"] == "1")
        ev_new = first(m["prepared_validation"], lambda r: r["label"] == "1" and r["key"] not in eval_keys
                       and (r["key"] in trainval) == (ev_old["key"] in trainval))
        pv0 = first(m["prepared_validation"])
        flip = dict(t0, label="1" if t0["label"] == "0" else "0", label_name=CELEBA_LABEL_NAME["1" if t0["label"] == "0" else "0"])
        faults = [
            ("G1-es-row-in-train", "first early-stopping row added to the CNN train", {"cnn_train": m["cnn_train"] + [es0]},
             {f"{P}05-official-split", f"{P}06-fit-vs-es", f"{P}14-declared-counts"}),
            ("G2-sha-swapped-in-cnn-test", "SHA-256 of the first two CNN test rows swapped (parquet and prepared manifests unchanged)",
             {"cnn_test": swap(m["cnn_test"], t0, t1)}, {f"{P}13-key-consistency"}),
            ("G3-label-name-out-of-domain", "label_name 'unknown' in the first CNN test row",
             {"cnn_test": [dict(r, label_name="unknown") if r is t0 else r for r in m["cnn_test"]]}, {ROW_VALIDATION}),
            ("G4-official-split-out-of-domain", "official_split 'train' in the first early-stopping row",
             {"cnn_es": [dict(r, official_split="train") if r is es0 else r for r in m["cnn_es"]]}, {ROW_VALIDATION}),
            ("G5-duplicated-pooled-test-row", "first pooled test row duplicated", {"pooled_test": m["pooled_test"] + [m["pooled_test"][0]]},
             {f"{P}01-internal-duplicate-keys", f"{P}02-exact-duplicate-images", f"{P}10-pooled-partition", f"{P}14-declared-counts"}),
            ("G6-test-hash-under-train-key", "the first CNN train row gets the SHA-256 of the first prepared test image",
             {"cnn_train": [dict(r, sha256=pv0["sha256"]) if r is first(m["cnn_train"]) else r for r in m["cnn_train"]]},
             {f"{P}07-train-side-vs-test-side", f"{P}13-key-consistency"}),
            ("G7-eval-row-replaced", "an evaluation attack replaced by another prepared test attack (same pooled side)",
             {"eval": [ev_new if r is ev_old else r for r in m["eval"]]}, {f"{P}03-conversion-rule"}),
            ("G8-label-flipped-consistently-in-row", "label and label_name of the first CNN test row flipped together (valid domain)",
             {"cnn_test": [flip if r is t0 else r for r in m["cnn_test"]]}, {f"{P}13-key-consistency", f"{P}14-declared-counts"}),
            ("G9-prepared-row-removed", "a prepared training row (not of the duplicate pair) removed",
             {"prepared_training": [r for r in m["prepared_training"] if r is not pt]},
             {f"{P}03-conversion-rule", f"{P}05-official-split", f"{P}14-declared-counts"}),
            ("G10-row-moved-fit-to-es", "first fitting row moved to the early stopping (partition and disjointness still hold)",
             {"cnn_train": [r for r in m["cnn_train"] if r is not first(m["cnn_train"])], "cnn_es": m["cnn_es"] + [first(m["cnn_train"])]},
             {f"{P}14-declared-counts"}),
        ]
    return faults


def run_self_test(ds: str, doc: dict, rules_p: Path, man_dir: Path, mans: dict, out: Path) -> bool:
    import subprocess
    import tempfile
    results = []
    with tempfile.TemporaryDirectory(prefix="m2-rgb-self-test-") as tmp:
        for fid, desc, changed, expected in self_test_faults(ds, mans):
            d = Path(tmp) / fid
            d.mkdir()
            rules = json.loads(json.dumps(doc))
            for n, m in rules["manifests"].items():
                q = d / m["file"]
                if n in changed:
                    write_manifest(ds, q, changed[n])          # forma canonica: il guasto deve arrivare alle regole
                    m["expected_rows"], m["expected_sha256"] = len(changed[n]), sha256(q)
                else:
                    q.write_bytes((man_dir / m["file"]).read_bytes())
            (d / "rules.json").write_text(json.dumps(rules, indent=1), encoding="utf-8")
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "check-overlap", "--dataset", ds, "--rules", str(d / "rules.json"),
                                "--manifests-dir", str(d), "--out", str(d / "out")], capture_output=True, text=True)
            rep_p = d / "out" / "overlap_report.json"
            rep = json.loads(rep_p.read_text()) if rep_p.exists() else {}
            failing = ([ROW_VALIDATION] if rep.get("row_validation_failed") else
                       sorted(v["id"] for v in rep.get("rules", []) if v["verdict"] == "FAIL"))
            checksum_stop = "differ from those declared" in r.stderr
            ok = r.returncode == 1 and set(failing) == expected and not checksum_stop and "Traceback" not in r.stderr
            results.append({"fault": fid, "description": desc, "changed_manifests": sorted(changed), "expected_failing_rules": sorted(expected),
                            "failing_rules": failing, "exit_code": r.returncode, "stopped_by_checksum": checksum_stop, "detected_as_expected": ok})
            print(f"   self-test {fid:<40} {'ok' if ok else 'NOT AS EXPECTED'} (failing: {', '.join(failing) or 'none'})", flush=True)
    passed = len(results) >= 8 and all(x["detected_as_expected"] for x in results)
    st_rep = {"dataset": ds, "rules_file": rel(rules_p), "rules_sha256": sha256(rules_p), "faults": results, "n_faults": len(results),
              "n_detected_as_expected": sum(x["detected_as_expected"] for x in results), "passed": passed,
              "note": "each faulty copy of the real manifests is checked with its rows and SHA-256 re-declared (so the checksum does not "
                      "stop it); it must fail exactly the expected rules, or the row validation for out-of-domain or inconsistent fields",
              **repo_state(), "checked": now()}
    (out / "overlap_self_test.json").write_text(json.dumps(st_rep, indent=1) + "\n", encoding="utf-8")
    L = [f"# {SPEC[ds]['title']} overlap self-test", "", f"**Result: {'PASS' if passed else 'FAIL'}** ({st_rep['n_detected_as_expected']} of "
         f"{len(results)} faulty copies rejected by exactly the expected rules).", "",
         "| fault | change | expected failing rules | failing rules | exit code | as expected |", "|---|---|---|---|---|---|"]
    L += [f"| {x['fault']} | {x['description']} | {', '.join(x['expected_failing_rules'])} | {', '.join(x['failing_rules']) or 'none'} | "
          f"{x['exit_code']} | {'yes' if x['detected_as_expected'] else 'NO'} |" for x in results]
    (out / "overlap_self_test.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return passed


# ============================================================================ recount e reconstruct

def cmd_recount(a) -> None:
    ds = a.dataset
    man_dir = Path(a.manifests_dir or ds_out(Path(a.out_root), ds) / "manifests")
    try:
        doc, mans, _ = load_declared(ds, Path(a.rules or rules_path(ds)), man_dir)
    except ManifestError as e:
        fail(str(e))
    agg = load_json(Path(a.aggregate or ds_out(Path(a.out_root), ds) / f"{ds}_manifests_aggregate.json"))
    fresh = recount_block(ds, mans)
    problems = [f"recount.{k} differs from the aggregate" for k, v in fresh.items() if (agg.get("recount") or {}).get(k) != v]
    for n, m in doc["manifests"].items():
        am = (agg.get("manifests") or {}).get(n, {})
        if am.get("sha256") != m["expected_sha256"]:
            problems.append(f"manifests.{n}.sha256 differs from the declared manifest")
        if {k: am.get(k) for k in fresh["manifest_counts"][n]} != fresh["manifest_counts"][n]:
            problems.append(f"manifests.{n} counts differ from the recount")
    po = fresh["pooled_overlap"]
    print("   " + "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in po.items()), flush=True)
    print(f"   exact duplicates by hash: {fresh['exact_duplicates_by_hash']}", flush=True)
    if problems:
        fail("recount differs from the aggregate: " + "; ".join(problems))
    print(f"== recount equal to the aggregate ({len(mans)} manifests, exclusions, pooled overlaps, duplicates)", flush=True)


def cmd_reconstruct(a) -> None:
    """Split official e pooled dai soli manifest (nessuna immagine). Una differenza con versioni di scikit-learn diverse
    da quelle dichiarate è segnalata come deriva della libreria, con un errore esplicito."""
    ds = a.dataset
    man_dir = Path(a.manifests_dir or ds_out(Path(a.out_root), ds) / "manifests")
    try:
        _, mans, _ = load_declared(ds, Path(a.rules or rules_path(ds)), man_dir)
    except ManifestError as e:
        fail(str(e))
    src_train, src_test = (("mirror_train", "mirror_test") if ds == "casia_fasd" else ("prepared_training", "prepared_validation"))
    smp = sampling(ds, mans[src_train], mans[src_test])
    sp = split_keys(ds, smp)
    problems = []
    for n in ("cnn_train", "cnn_es", "cnn_test", "pooled_train", "pooled_es", "pooled_test"):
        have = sorted(r["key"] for r in mans[n])
        same = sorted(sp[n]) == have
        print(f"   {n}: rebuilt {len(sp[n]):,}, manifest {len(have):,}, identical: {'yes' if same else 'NO'}", flush=True)
        if not same:
            problems.append(n)
    if ds == "casia_fasd":
        u = held_out_uniqueness(mans["cnn_train"] + mans["cnn_es"], len(mans["cnn_es"]), 4, mans["cnn_es"])
        print(f"   independent of scikit-learn: {u['n_with_card_n_val']} of {u['n_combinations']:,} combinations of 4 of the "
              f"{u['n_training_subjects']} training subjects give {len(mans['cnn_es'])} early-stopping images, "
              f"{u['n_with_card_n_val_and_class_counts']} also with the same counts per class: the held-out subjects are not "
              "pinned by the counts alone (the declared split relies on GroupShuffleSplit, seed 42)", flush=True)
    if problems:
        v = library_versions()
        hint = (f" The declared manifests were written with {BUILT_WITH}; this environment has {v}: the difference may be a "
                "drift of the random split of the library rather than a change of the data (this command reports the possibility, it "
                "does not establish it). The declared manifests remain the reference."
                if v != BUILT_WITH else "")
        fail(f"the splits rebuilt from the manifests differ from {problems}.{hint}")
    print(f"== official and pooled splits rebuilt from {src_train} and {src_test} alone: identical (no image read)", flush=True)


# ============================================================================ report

def report_fields(ds: str, summ: dict, ov: dict, private: bool, self_test: dict = None) -> dict:
    m = summ["manifests"]
    if ds == "casia_fasd":
        head = "| manifest | file | images | bona fide | attack | subjects | videos | SHA-256 |\n|---|---|---|---|---|---|---|---|"
        rows = [f"| {n} | `{x['file']}` | {x['n']:,} | {x['n_bona_fide']:,} | {x['n_attack']:,} | {x['n_subjects']} | {x['n_videos']} | `{x['sha256'][:16]}…` |"
                for n, x in m.items()]
    else:
        head = "| manifest | file | images | bona fide | attack | SHA-256 |\n|---|---|---|---|---|---|"
        rows = [f"| {n} | `{x['file']}` | {x['n']:,} | {x['n_bona_fide']:,} | {x['n_attack']:,} | `{x['sha256'][:16]}…` |" for n, x in m.items()]
    ck = [f"| {n} | {c['protocol']} | `{c['weights_sha256'][:16]}…` | {'yes' if c['weights_sha256_verified'] else 'NO'} | "
          f"{c['card_counts']['n_train']:,}/{c['card_counts']['n_val']:,}/{c['card_counts']['n_test']:,} | {'yes' if c['counts_match_manifests'] else 'NO'} | {c['val_split']} |"
          for n, c in summ["checkpoints"].items()]
    verdicts = [f"| {v['id']} | {v['severity']} | {', '.join(v['by']) or '-'} | **{v['verdict']}** | {v['description']} |" for v in ov["rules"]]
    notes = []
    for v in ov["rules"]:
        for c in v["checks"]:
            if c["outcome"] == NOT_CHECKABLE:
                notes.append(f"- {v['id']} (not checkable): {c['reason']}")
            elif "expected" in c and "n_shared" in c:
                notes.append(f"- {v['id']}: {c['a']} vs {c['b']}, by {c['by']}: {c['n_shared']:,} shared (declared {c['expected']:,}) → {c['outcome']}")
            elif "expected" in c and "duplicated_values" in c and c["expected"]:
                notes.append(f"- {v['id']}: {c['manifest']}: {plural(c['duplicated_values'], 'duplicated ' + c['by'] + ' value')} ({c['rows']} rows; declared {c['expected']}) → {c['outcome']}")
            elif c["outcome"] == "reported":
                notes.append(f"- {v['id']}: {c.get('a', c.get('manifest'))} vs {c.get('b', '-')}, by {c['by']}: {c.get('n_shared', c.get('duplicated_values'))} shared values (reported)")
    rc = summ["reconstruction"]
    der = (rc.get("training_cache") or {}).get("derived_frames")
    derived = ("not checked: the training cache is missing" if not der else
               f"train {der['train']['n_frames']:,} frames ({der['train']['distinct']:,} distinct, {der['train']['duplicated_values']} duplicated values); "
               f"test {der['test']['n_frames']:,} ({der['test']['distinct']:,} distinct, {der['test']['duplicated_values']} duplicated values); "
               f"shared between train and test {der['train_test_shared']}, between fitting and early-stopping images {der['fit_vs_validation_shared']}, "
               f"between the pooled training side and the pooled test {der['pooled_trainval_vs_test_shared']}. {der['linkage'][0].upper() + der['linkage'][1:]}.")
    if ds == "casia_fasd":
        u = rc.get("held_out_uniqueness") or {}
        recon = (f"Official split: {rc['official']}. Independently of the scikit-learn version, {u.get('n_with_card_n_val')} of the "
                 f"{u.get('n_combinations'):,} combinations of 4 of the {u.get('n_training_subjects')} training subjects give the "
                 f"{m['cnn_es']['n']:,} early-stopping images of the cards ({u.get('n_with_card_n_val_and_class_counts')} also with the "
                 "same counts per class): the counts do not pin the held-out subjects, so the reconstruction relies on "
                 "`GroupShuffleSplit` with seed 42; it is verified against the card counts and the label and group arrays of the cache.")
        if private:
            recon += f" Early-stopping subjects: {', '.join(summ['restricted_records']['early_stopping_subjects'])}."
    else:
        alt = rc.get("alternative_hypothesis") or {}
        vb = alt.get("val_by_class") or {}
        recon = (f"Official split: {rc['official']}. Alternative: {alt.get('split')} gives {alt.get('n_train'):,} / {alt.get('n_val'):,} "
                 f"({vb.get('bona_fide')} bona fide and {vb.get('attack')} attacks in the early stopping); {alt.get('note')}.")
    recon += f" Pooled split: {rc['pooled']}. Sampling: {rc['sampling']}. Logic: {rc['logic']}."
    src = summ["source"]
    if ds == "casia_fasd":
        source = (f"{src['mirror']}, revision `{src['revision']}`; archive `{src['archive']}` ({src['archive_bytes']:,} bytes), SHA-256 "
                  f"`{src['archive_sha256']}` (equal to the etag of the Hugging Face download cache: {'yes' if src['archive_sha256_equals_hf_etag'] else 'NO'}).")
    else:
        source = (f"{src['mirror']}, revision `{src['revision']}`; file `{src['file']}` ({src['file_bytes']:,} bytes, {src['shard']}), SHA-256 "
                  f"`{src['file_sha256']}` (equal to the etag of the Hugging Face download cache: {'yes' if src['file_sha256_equals_hf_etag'] else 'NO'}); "
                  f"{src['rows']:,} rows, {src['rows_without_image']} without image bytes ({src['rows_without_image_by_label'].get('bona_fide', 0)} live, "
                  f"{src['rows_without_image_by_label'].get('attack', 0)} spoof).")
        if private:
            source += f" Rows without image: {summ['restricted_records']['rows_without_image']}; exact duplicate rows: {summ['restricted_records']['exact_duplicate_rows']}."
    if self_test:
        st_line = (f"{'PASS' if self_test.get('passed') else 'FAIL'}, {self_test.get('n_detected_as_expected')} of {self_test.get('n_faults')} faulty copies "
                   "rejected by exactly the expected rules (" + "; ".join(f"{x['fault']}: {', '.join(x['failing_rules']) or 'none'}" for x in self_test["faults"]) + ").")
    else:
        st_line = "not run (`check-overlap --self-test` writes `overlap_self_test.json`)."
    return {"title": SPEC[ds]["title"], "dataset": ds, "commit": summ.get("commit"), "created": summ.get("created"),
            "visibility": ("Restricted version: it may name subject keys and row indices." if private
                           else "Public version: no subject, video or row identifiers and no per-image data."),
            "source": source, "declarations": "\n".join(f"- {k}: {v}" for k, v in SOURCE_DECLARATIONS[ds]),
            "manifest_table": head + "\n" + "\n".join(rows), "checkpoint_rows": "\n".join(ck), "verdict_rows": "\n".join(verdicts),
            "overlap_result": "PASS" if ov["passed"] else "FAIL", "n_rules": len(ov["rules"]), "n_must_failed": ov["n_must_failed"],
            "rule_notes": "\n".join(notes) or "- none", "reconstruction": recon, "derived": derived, "self_test_line": st_line,
            "limits": "\n".join(f"- {x}" for x in LIMITS[ds]), "supported": "\n".join(f"- {x}" for x in SUPPORTED[ds]),
            "build_passed": "yes" if summ.get("passed") else "NO", "n_checks": len(summ.get("checks", {})),
            "sha_identity": (f"The SHA-256 of every manifest is identical to the SHA-256 declared in `{ds}_overlap_rules.json`."
                             if (summ.get("checks") or {}).get("manifests_equal_declared_in_rules") else
                             f"**The manifests differ from those declared in `{ds}_overlap_rules.json`.**")}


def cmd_report(a) -> None:
    ds = a.dataset
    out = Path(a.out or ds_out(Path(a.out_root), ds))
    summ = load_json(out / f"{ds}_manifests_summary.json")
    agg = load_json(out / f"{ds}_manifests_aggregate.json")
    ov = load_json(out / "overlap_report.json")
    if summ.get("passed") is not True or not (summ.get("checks") or {}).get("manifests_equal_declared_in_rules"):
        fail("the summary records failed build checks (or manifests different from the declared ones): no report is written")
    if {n: v["sha256"] for n, v in agg["manifests"].items()} != {n: v["sha256"] for n, v in summ["manifests"].items()}:
        fail("aggregate and summary list different manifests: rerun build")
    by_file = {v["file"].split("/")[-1]: v["sha256"] for v in summ["manifests"].values()}
    stale = [n for n, v in (ov.get("manifests") or {}).items() if by_file.get(v["file"]) != v["sha256"]]
    if stale or ov.get("row_validation_failed") or not ov.get("manifests"):
        fail(f"overlap_report.json does not refer to the manifests of the summary ({stale or 'no manifests'}): rerun check-overlap")
    st_p = out / "overlap_self_test.json"
    self_test = load_json(st_p) if st_p.is_file() else None
    if self_test is not None and self_test.get("rules_sha256") != ov.get("rules_sha256"):
        fail("overlap_self_test.json was written with different rules than overlap_report.json: rerun check-overlap --self-test")
    tmpl = Path(a.template).read_text(encoding="utf-8")
    mans = {}
    for n, v in summ["manifests"].items():
        p = out / v["file"]
        if p.is_file():
            mans[n] = read_csv(p, SPEC[ds]["columns"])
    ids = identifiers(ds, mans)
    for private, fname in ((True, f"{ds}_m2_report.md"), (False, f"{ds}_m2_report_public.md")):
        txt = tmpl.format(**report_fields(ds, summ, ov, private, self_test))
        if not private:
            if not mans:
                fail("the manifests are needed to scan the public report for identifiers")
            bad = public_text_problems(txt, ids, markdown=True)
            if bad:
                fail("public report contains forbidden content: " + "; ".join(bad))
        (out / fname).write_text(txt, encoding="utf-8")
        print(f"== {relpath(out / fname)} sha256 {sha256(out / fname)}", flush=True)


# ============================================================================ CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, out=True):
        p.add_argument("--dataset", required=True, choices=DATASETS)
        p.add_argument("--out-root", default=str(OUT_DIR), help="cartella radice dei risultati (default results/m2_rgb; <radice>/<dataset>)")
        if out:
            p.add_argument("--out", default=None, help="cartella del dataset (default <out-root>/<dataset>)")

    p = sub.add_parser("build", help="manifest per immagine, riepilogo privato e aggregato pubblico")
    common(p)
    p = sub.add_parser("check-overlap", help="verdetto delle regole dichiarate (codice 1 se una regola must fallisce)")
    common(p)
    p.add_argument("--rules", default=None, help="file delle regole (default delivery/m2/rgb/<dataset>_overlap_rules.json)")
    p.add_argument("--manifests-dir", default=None, help="cartella dei manifest (default <out-root>/<dataset>/manifests)")
    p.add_argument("--self-test", action="store_true", help="ripete il controllo su copie guaste dei manifest reali")
    p = sub.add_parser("recount", help="ricalcola dai soli manifest il blocco ricontabile dell'aggregato")
    common(p, out=False)
    p.add_argument("--rules", default=None)
    p.add_argument("--manifests-dir", default=None)
    p.add_argument("--aggregate", default=None, help="aggregato pubblico da verificare")
    p = sub.add_parser("reconstruct", help="ricostruisce gli split dai soli manifest (scikit-learn, nessuna immagine)")
    common(p, out=False)
    p.add_argument("--rules", default=None)
    p.add_argument("--manifests-dir", default=None)
    p = sub.add_parser("report", help="rapporto breve privato e pubblico")
    common(p)
    p.add_argument("--template", default=str(REPORT_TMPL))
    a = ap.parse_args()
    {"build": cmd_build, "check-overlap": cmd_check_overlap, "recount": cmd_recount, "reconstruct": cmd_reconstruct,
     "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
