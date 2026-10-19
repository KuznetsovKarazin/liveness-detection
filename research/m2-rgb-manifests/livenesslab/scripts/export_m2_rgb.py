"""
Esporta il pacchetto della scheda M2, seconda tranche (CASIA-FASD e CelebA-Spoof), in una cartella pulita: parte
pubblica research/m2-rgb-manifests/ e parte riservata drive/02_Experiments/M2/<run-id>/. Fratello di
scripts/export_m2.py (tranche NUAA, invariato), di cui riusa scansione, completezza, SHA256SUMS e campi di ambiente
tramite scripts/export_helpers.py (copia testuale verificata delle funzioni di export_m1.py ed export_m2.py).

Passi:
  1. rigenera in --m2 (default results/m2_rgb) build, check-overlap --self-test, recount, reconstruct e report di
     scripts/rgb_manifests.py per i due dataset al commit corrente, poi i test (check_rgb_manifests.py --require-real,
     check_nuaa_manifests.py, check_export_m2_rgb.py); i log vanno nella cartella riservata con i percorsi resi relativi;
  2. parte pubblica (ricreata vuota): README.md (da delivery/m2/rgb/README.md con commit, run-id e stato di consegna),
     naming_schema_rgb.md, transformations_rgb.md, le due regole, per dataset aggregato, rapporto delle sovrapposizioni
     pubblico e rapporto breve pubblico, run-report.md, livenesslab/ (elenco esplicito CODE), SHA256SUMS per ultimo; i
     test offline rieseguiti dalla copia in livenesslab/ con cartelle dati e risultati vuote;
  3. parte riservata (ricreata vuota): <dataset>/manifests, riepilogo, rapporti riservati, self-test, rapporto privato;
     registri dei dati (bozze), log, pip_freeze.txt, NOT_PRODUCED.md, public_SHA256SUMS.sha256 (ancora), README.md,
     run-report.md (copia), SHA256SUMS per ultimo;
  4. scansione del contenuto vietato su entrambe le cartelle e, nella pubblica, di chiavi di soggetto e video, nomi di
     frame, chiavi di riga e SHA-256 delle immagini dei manifest; completezza delle due cartelle.

Il run-id predefinito è AAAAMMGG-M2-rgb-seed42-<commit>, distinto da quello della tranche NUAA (la cartella riservata
di un run NUAA dello stesso giorno non viene mai toccata). Con --strict l'export si rifiuta di partire se il codice o i
documenti esportati hanno modifiche non registrate. Lo stato del repository nei JSON (commit, dirty) esclude la cartella
--m2 che l'export cancella e rigenera (dirty_excludes): l'export verifica che durante la rigenerazione nulla cambi fuori
da --m2 e che ogni JSON registri lo stesso stato del run report. Ogni test saltato è un controllo fallito, con il motivo:
nessun test salta per la piattaforma, quindi un salto è sempre un ingresso mancante. Codice 1 se un controllo fallisce. Non scrive mai in data, models,
results/c1, results/eval, results/m1 o results/m2 (tranche NUAA).
Uso:  python scripts/export_m2_rgb.py [--out export] [--m2 results/m2_rgb] [--run-id ...] [--strict] [--dry-run]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# funzioni di servizio di export_m1.py ed export_m2.py, copiate testualmente in export_helpers.py (pubblicabile); il test
# check_export_m2_rgb.py ne verifica l'identità con gli originali nel repository di sviluppo
from export_helpers import (completeness_problems, env_versions, fill_run_id, hardware_line, local_imports, redact,  # noqa: E402
                            run, scan, sha256, tests_line, write_sums)
from package_rules import local_patterns_problem  # noqa: E402
import path_guard  # noqa: E402
import rgb_manifests as RM  # noqa: E402
from rgb_manifests import AUTHORISATION_NOTE  # noqa: E402

PY = sys.executable
ROME = ZoneInfo("Europe/Rome")      # data del run-id e del run report nel fuso del progetto, qualunque sia il TZ della macchina
DELIV = ROOT / "delivery" / "m2" / "rgb"
PUBLIC_NAME = "m2-rgb-manifests"
PUBLIC_ANCHOR = "public_SHA256SUMS.sha256"
DATASETS = ("casia_fasd", "celeba_spoof")
REVIEW_LINE = "not yet reviewed (first delivery of this tranche)."
RUN_ID_RX = re.compile(r"^\d{8}-M2-rgb-[A-Za-z0-9._-]+$")
STRICT_PATHS = ["scripts", "src", "delivery", "requirements*.txt"]
READ_ONLY = ["results/c1", "results/eval", "results/m1", "results/m2", "models", "data"]
# solo requirements.txt: nessun comando ne usa altri
CODE = ["requirements.txt", "src/tesi_app/__init__.py", "src/tesi_app/paths.py",
        "scripts/rgb_manifests.py", "scripts/check_rgb_manifests.py", "scripts/nuaa_manifests.py", "scripts/select_threshold.py",
        "scripts/package_rules.py", "delivery/m2/rgb/casia_fasd_overlap_rules.json", "delivery/m2/rgb/celeba_spoof_overlap_rules.json",
        "delivery/m2/rgb/report.md.tmpl", "delivery/m2/rgb/README.md", "delivery/m2/rgb/run-report.md.tmpl",
        # script di controllo dell'inventario pubblicato sulla scheda (promessi per questa PR): rigenerano i 71 numeri
        "scripts/m2_inventory/common.py", "scripts/m2_inventory/casia_inventory.py", "scripts/m2_inventory/celeba_inventory.py",
        "scripts/m2_inventory/splits_inventory.py", "scripts/m2_inventory/checkpoints_inventory.py", "scripts/m2_inventory/run_all.py",
        # export, suoi test e controllo delle mutazioni, da rieseguire dal pacchetto pubblico (richiesta del coordinatore)
        "scripts/export_m2_rgb.py", "scripts/export_helpers.py", "scripts/check_export_m2_rgb.py", "scripts/check_rgb_mutations.py",
        # protezione delle cartelle da cancellare, comune agli export (16/10/2026)
        "scripts/path_guard.py"]
PUBLIC_DOCS = ["naming_schema_rgb.md", "transformations_rgb.md", "casia_fasd_overlap_rules.json", "celeba_spoof_overlap_rules.json"]
# per dataset: uscita di rgb_manifests.py in <m2>/<dataset>/ -> nome nel pacchetto pubblico
PUBLIC_RESULTS = {"{ds}_manifests_aggregate.json": "{ds}_manifests_aggregate.json",
                  "overlap_report_public.json": "{ds}_overlap_report_public.json",
                  "overlap_report_public.md": "{ds}_overlap_report_public.md",
                  "{ds}_m2_report_public.md": "{ds}_report.md"}
RESTRICTED_RESULTS = ["{ds}_manifests_summary.json", "overlap_report.json", "overlap_report.md", "overlap_self_test.json",
                      "overlap_self_test.md", "{ds}_m2_report.md"]
RESTRICTED_DOCS = ["dataset_register_casia_fasd.md", "dataset_register_celeba_spoof.md"]
# file condivisi con la tranche NUAA: identici a quelli consegnati (SHA-256 dei file pubblicati)
REFERENCE_FILES = {
    "scripts/nuaa_manifests.py": "research/m2-nuaa-manifests/livenesslab/scripts/nuaa_manifests.py",
    "scripts/select_threshold.py": "research/m2-nuaa-manifests/livenesslab/scripts/select_threshold.py",
    "src/tesi_app/paths.py": "research/m2-nuaa-manifests/livenesslab/src/tesi_app/paths.py",
    "scripts/package_rules.py": "research/m2-nuaa-manifests/livenesslab/scripts/package_rules.py",
}
DESCR = {
    "casia_fasd": "CASIA-FASD: `manifests/` (11 per-image manifests), private summary, restricted overlap report, self-test, private report",
    "celeba_spoof": "CelebA-Spoof: `manifests/` (10 per-image manifests), private summary, restricted overlap report, self-test, private report",
    "dataset_register_casia_fasd.md": "draft entry of the dataset register (01_Dataset_Register) for CASIA-FASD",
    "dataset_register_celeba_spoof.md": "draft entry of the dataset register (01_Dataset_Register) for CelebA-Spoof, with the personal-data note",
    "run-report.md": "copy of the run report of the public package",
    "pip_freeze.txt": "exact package versions of the environment",
    "NOT_PRODUCED.md": "artefacts not produced in this run, with the reason",
    PUBLIC_ANCHOR: f"SHA-256 of the `SHA256SUMS` of the public package `research/{PUBLIC_NAME}/` (`shasum -a 256` format)",
    "SHA256SUMS": "SHA-256 of every file of this folder and of its subfolders, except itself",
}


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def published_sha(path: str):
    """SHA-256 del file pubblicato nel pacchetto NUAA, letto dal suo SHA256SUMS in export/ (None se non disponibile)."""
    sums = ROOT / "export" / "research" / "m2-nuaa-manifests" / "SHA256SUMS"
    if not sums.is_file():
        return None
    sub = path.split("research/m2-nuaa-manifests/", 1)[1]
    for line in sums.read_text(encoding="utf-8").splitlines():
        h, _, f = line.partition("  ")
        if f == sub:
            return h
    return None


def copy_code(dst_root: Path) -> list:
    missing = []
    for item in CODE:
        src, dst = ROOT / item, dst_root / item
        if not src.is_file():
            missing.append(item); continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return missing


def code_import_problems(code_root: Path) -> list:
    listed = {Path(i).stem for i in CODE if i.startswith("scripts/")}
    return [f"{py.name} imports {sorted(extra)} from scripts/, not in the code list"
            for py in sorted((code_root / "scripts").glob("*.py")) if (extra := local_imports(py) - listed)]


def record_ids(d: Path) -> dict:
    """Identificativi reali dai manifest riservati (chiavi di soggetto e video, nomi, chiavi di riga, SHA-256)."""
    import rgb_manifests as RM
    ids = {"subject_keys": set(), "video_keys": set(), "names": set(), "hashes": set()}
    for ds in DATASETS:
        man = d / ds / "manifests"
        if not man.is_dir():
            continue
        mans = {p.stem: RM.read_csv(p, RM.SPEC[ds]["columns"]) for p in sorted(man.glob("*.csv"))}
        for k, v in RM.identifiers(ds, mans).items():
            ids[k] |= v
    return ids


def public_record_problems(folder: Path, ids: dict) -> list:
    import rgb_manifests as RM
    bad = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and not p.is_symlink():
            for r in RM.record_problems(p.read_bytes().decode("utf-8", errors="replace"), ids):
                bad.append((p.relative_to(folder).as_posix(), r))
    return bad


def today() -> str:
    return datetime.now(ROME).strftime("%Y%m%d")


def delivery_status(run_id: str, up_run=None, up_on=None, up_sums=None, m2: Path = None) -> tuple:
    """Stato di consegna del README: cartella pronta in locale e, se l'operatore dichiara un caricamento precedente,
    confronto dei manifest (<dataset>/manifests/*.csv) con lo SHA256SUMS di quella cartella. Restituisce (testo, problemi)."""
    here = (f"The restricted folder of this run (`{run_id}`) is kept locally and not uploaded. The provenance note "
            "`M2_provenienza_RGB_20261009.pdf` was placed on 9 October 2026 in the shared M2 folder `20261006-M2-seed42-36c677a` as a "
            "supplementary document outside the NUAA run (not listed in its `SHA256SUMS`), as indicated in the coordinator's message "
            "of 9 October 2026 (email and M2 card); the restricted RGB run folder stays local and is uploaded to "
            f"`02_Experiments/M2/{run_id}/` only when he asks for it.")
    if not up_run:
        return here + " No earlier upload of a CASIA-FASD or CelebA-Spoof restricted folder is declared.", []
    listed = {}
    for line in Path(up_sums).read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"([0-9a-f]{64})  ((?:casia_fasd|celeba_spoof)/manifests/[^/]+\.csv)", line)
        if m:
            listed[m.group(2)] = m.group(1)
    current = {f"{ds}/manifests/{p.name}": sha256(p) for ds in DATASETS for p in sorted((Path(m2) / ds / "manifests").glob("*.csv"))}
    if not listed or listed != current:
        diff = sorted(k for k in set(listed) | set(current) if listed.get(k) != current.get(k))
        return here, [f"manifests differ from the uploaded restricted folder {up_run}: {diff[:5] or 'no manifest listed'}"]
    return (f"Restricted folder `02_Experiments/M2/{up_run}/` uploaded to Drive on {up_on} (declared by the operator): "
            f"the {len(current)} manifests are byte-identical to that upload (SHA-256 compared with its `SHA256SUMS`); other files are "
            "not compared. " + here), []


def artefact_problems(folder: Path, ds: str, commit: str) -> list:
    """Artefatti di un dataset non pubblicabili: mancanti, falliti, di un altro commit, incoerenti fra loro (manifest di
    aggregato, riepilogo e rapporto delle sovrapposizioni; regole del self-test), o file temporanei rimasti."""
    folder = Path(folder)
    problems = [f"{ds}: leftover temporary file {p.name}" for p in sorted(folder.glob("*.tmp"))]

    def doc(name):
        try:
            return json.loads((folder / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            problems.append(f"{ds}: {name} missing or unreadable")
            return {}
    summ, agg = doc(f"{ds}_manifests_summary.json"), doc(f"{ds}_manifests_aggregate.json")
    ov, ovp, st = doc("overlap_report.json"), doc("overlap_report_public.json"), doc("overlap_self_test.json")
    for name, d in ((f"{ds}_manifests_summary.json", summ), (f"{ds}_manifests_aggregate.json", agg), ("overlap_report.json", ov),
                    ("overlap_report_public.json", ovp), ("overlap_self_test.json", st)):
        if d and (d.get("passed") is not True or d.get("commit") != commit):
            problems.append(f"{ds}: {name} is not a passed artefact of commit {commit[:7]}")
    shas = lambda m: {v["file"].split("/")[-1]: v["sha256"] for v in (m or {}).values()}   # noqa: E731
    if summ and agg and shas(summ.get("manifests")) != shas(agg.get("manifests")):
        problems.append(f"{ds}: aggregate and summary list different manifests")
    for name, d in (("overlap_report.json", ov), ("overlap_report_public.json", ovp)):
        if summ and d and shas(d.get("manifests")) != shas(summ.get("manifests")):
            problems.append(f"{ds}: {name} refers to other manifests than the summary")
    if st and ov and st.get("rules_sha256") != ov.get("rules_sha256"):
        problems.append(f"{ds}: self-test written with other rules than the overlap report")
    for name in (f"{ds}_m2_report.md", f"{ds}_m2_report_public.md", "overlap_report.md", "overlap_report_public.md"):
        if not (folder / name).is_file():
            problems.append(f"{ds}: {name} missing")
    return problems


# ----------------------------------------------------------------------------- protezione dei percorsi
# Cartelle che l'export non deve mai cancellare né sovrascrivere, né contenere in una cartella da cancellare
# (correzione del 16/10/2026: un --m2 antenato di una cartella protetta, per esempio results/ o la radice del
# repository, passava il controllo; poi il confronto per identità dei file, perché su un filesystem che non distingue le
# maiuscole una grafia diversa dello stesso percorso passava un confronto fra stringhe). La logica è in path_guard.py.
PROTECTED = [x for x in path_guard.BASE_PROTECTED if x != "results/m2_rgb"]


def allowed_folders(run_id: str, root: Path = None) -> list:
    """Le sole cartelle del repository che questo export ricrea: results/m2_rgb, il pacchetto pubblico della tranche e la
    cartella riservata del run se non esiste già o è di una prova (-trial): un run consegnato non si riscrive."""
    root = Path(root if root is not None else ROOT)        # ROOT letto alla chiamata (i test usano un albero finto)
    run_dir = root / "export" / "drive" / "02_Experiments" / "M2" / run_id
    allowed = [root / "results" / "m2_rgb", root / "export" / "research" / PUBLIC_NAME]
    if run_id and (not run_dir.exists() or run_id.endswith("-trial")):
        allowed.append(run_dir)
    return allowed


def deletion_problem(path, run_id: str, root: Path = None, home: Path = None) -> str:
    """Motivo per cui una cartella non si può cancellare o riscrivere ("" se si può), con le regole di path_guard.py."""
    root = root if root is not None else ROOT
    return path_guard.deletion_problem(path, root, PROTECTED, allowed_folders(run_id, root), home)


def safe_rmtree(path, run_id: str) -> None:
    """Cancella una cartella dell'export solo dopo il controllo di deletion_problem (subito prima della cancellazione)."""
    msg = deletion_problem(path, run_id)
    if msg:
        raise SystemExit(f"ERROR: refusing to delete: {msg}")
    shutil.rmtree(path, ignore_errors=True)


def nuaa_package_problem(root: Path) -> str:
    """Il pacchetto pubblico NUAA serve per confrontare i file condivisi; con --strict la sua assenza ferma l'export."""
    if not (Path(root) / "export" / "research" / "m2-nuaa-manifests" / "SHA256SUMS").is_file():
        return "the NUAA public package (export/research/m2-nuaa-manifests) is missing: the shared files cannot be compared with it"
    return ""


TRIAL_NOTE = ("**Trial export, not for delivery**: built from a working tree with uncommitted changes (`--trial`); the run-id "
              "ends with `-trial` and the run report records `working tree clean at the start of the export: NO`.")


def effective_run_id(run_id: str, dirty: bool, trial: bool) -> str:
    """Run-id di una prova da albero modificato: suffisso `-trial`, così non si confonde con un run da consegnare."""
    return run_id + "-trial" if dirty and trial and not run_id.endswith("-trial") else run_id


def tree_gate(dirty: bool, trial: bool) -> str:
    """Un export da un albero con modifiche non registrate è ammesso solo come prova esplicita (--trial)."""
    if dirty and not trial:
        return ("the working tree has uncommitted changes: commit them first, or pass --trial for an explicit trial export "
                "(its run report records 'working tree clean at the start of the export: NO')")
    return ""


def yes_no(v) -> str:
    return {True: "yes", False: "NO"}.get(v, "not checked")


# ----------------------------------------------------------------------------- stato del repository (17/10/2026)
# I JSON di rgb_manifests.py registrano commit e dirty senza la cartella dei risultati che l'export cancella e rigenera
# (dirty_excludes): con results/ versionato quei file risultano cancellati e riscritti durante il run. L'export verifica
# che fuori da quella cartella nulla cambi durante la rigenerazione e che commit e dirty dei JSON coincidano con lo stato
# registrato nel run report, così i due resoconti non possono più divergere.
STATE_JSON = ["{ds}_manifests_summary.json", "{ds}_manifests_aggregate.json", "overlap_report.json", "overlap_report_public.json",
              "overlap_self_test.json"]


def tree_outside(m2) -> tuple:
    """(cartella esclusa o None, commit, git status fuori da essa): lo stesso calcolo dei JSON (rgb_manifests.tree_state)."""
    return RM.tree_state(m2)


def repo_dirty(root: Path = None) -> bool:
    """Albero con modifiche (file non versionati compresi) per la soglia iniziale: lo stesso calcolo dei JSON, senza
    esclusioni (tree_state della radice), così la configurazione di git dell'utente non può farli divergere."""
    root = Path(root if root is not None else ROOT)
    _, commit, status = RM.tree_state(root, root)
    return bool(commit) and (status is None or bool(status))     # git status fallito: non dimostrato pulito


def status_failed(state: tuple) -> bool:
    """git risponde (commit leggibile) ma git status fallisce: stato sconosciuto, mai letto come pulito."""
    return bool(state[1]) and state[2] is None


def tree_change_problem(before: tuple, after: tuple) -> str:
    """Una modifica fuori dalla cartella rigenerata, comparsa durante la rigenerazione, è un controllo fallito."""
    if status_failed(before) or status_failed(after):
        return "git status failed: the repository state outside the results folder is unknown"
    if before[1] != after[1] or (before[2] or "") != (after[2] or ""):
        changed = sorted(set((after[2] or "").splitlines()) ^ set((before[2] or "").splitlines()))
        return (f"the working tree changed outside {before[0] or 'the results folder'} during the regeneration "
                f"(commit {str(before[1])[:7]} -> {str(after[1])[:7]}; {changed[:5]})")
    return ""


def json_state_problems(m2: Path, commit: str, dirty: bool, excludes) -> list:
    """Ogni JSON rigenerato deve registrare il commit del run, dirty uguale allo stato fuori dalla cartella esclusa
    all'inizio dell'export e la stessa cartella esclusa; altrimenti JSON e run report direbbero cose diverse."""
    problems = []
    for ds in DATASETS:
        for n in STATE_JSON:
            name = n.format(ds=ds)
            try:
                doc = json.loads((Path(m2) / ds / name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue                                   # mancante o illeggibile: già segnalato da artefact_problems
            got = (doc.get("commit"), doc.get("dirty"), doc.get("dirty_excludes", "absent"))
            if got != (commit, dirty, excludes):
                problems.append(f"{ds}/{name}: repository state (commit {str(got[0])[:7]}, dirty {got[1]}, excludes {got[2]}) differs "
                                f"from the run report (commit {commit[:7]}, dirty {dirty}, excludes {excludes})")
    return problems


def tree_line(dirty_start: bool, excludes, json_dirty: bool) -> str:
    """Stato dell'albero nel run report, con la semantica esatta del campo dirty dei JSON."""
    start = "yes" if not dirty_start else "NO (explicit trial export, `--trial`)"
    if excludes:
        return (f"working tree clean at the start of the export: {start}; during the run only `{excludes}` was deleted and regenerated, "
                f"`git status` outside that folder was unchanged (checked after the regeneration), and the `dirty` field of the regenerated "
                f"JSON files excludes that folder (`dirty_excludes`) and is `{str(json_dirty).lower()}`")
    return (f"working tree clean at the start of the export: {start}; the results folder is outside the repository and the "
            f"`dirty` field of the regenerated JSON files is the plain repository state, `{str(json_dirty).lower()}`")


def skip_problems(name: str, txt: str, skipped) -> list:
    """Criterio di accettazione dei test nell'export: nessun test saltato. Dal 17/10/2026 nessun test salta per la
    piattaforma (il filesystem che distingue le maiuscole è simulato): uno skip è sempre un ingresso mancante (dati,
    registri, pacchetto NUAA, export di sviluppo) e per l'export ufficiale resta un controllo fallito, con il motivo."""
    if skipped is None:
        return [f"{name}: test summary missing"]
    if skipped:
        reasons = [l.split("skipped ", 1)[1] for l in txt.splitlines() if l.startswith("skipped ")]
        return [f"{name}: {skipped} skipped tests (missing inputs: {'; '.join(reasons) or 'reason not printed'})"]
    return []


def report_fields(summ: dict, ov: dict, st: dict, agg: dict, run_id, commit, a, m2: Path, counts_files, logs_info, failed, not_produced) -> dict:
    """Campi del run report dai file scritti da rgb_manifests.py (nessun numero scritto a mano)."""
    def man_line(ds):
        s = summ.get(ds) or {}
        return "; ".join(f"`{ds}/{v['file']}` `{v['sha256']}` ({v['n']:,})" for v in (s.get("manifests") or {}).values()) or f"{ds}: not produced"

    def counts(ds):
        m = (summ.get(ds) or {}).get("manifests") or {}
        if not m:
            return "not produced"
        g = lambda n: f"{m[n]['n']:,} ({m[n]['n_bona_fide']:,} + {m[n]['n_attack']:,})" if n in m else "n/a"   # noqa: E731
        if ds == "casia_fasd":
            return (f"mirror train {g('mirror_train')}, mirror test {g('mirror_test')}; official fit {g('cnn_train')} ({m['cnn_train']['n_subjects']} subjects), "
                    f"early stopping {g('cnn_es')} ({m['cnn_es']['n_subjects']} subjects), test {g('cnn_test')}; pooled {g('pooled_train')} / "
                    f"{g('pooled_es')} / {g('pooled_test')}; evaluation set {g('eval')}")
        return (f"parquet rows with image {g('parquet')}; prepared training {g('prepared_training')}, prepared validation {g('prepared_validation')}; "
                f"\"official\" fit {g('cnn_train')}, early stopping {g('cnn_es')}, test {g('cnn_test')}; pooled {g('pooled_train')} / {g('pooled_es')} / "
                f"{g('pooled_test')}; evaluation set {g('eval')}")

    def overlap(ds):
        o = ov.get(ds)
        if not o:
            return "not produced"
        po = ((agg.get(ds) or {}).get("recount") or {}).get("pooled_overlap") or {}
        nc = [v["id"] for v in o["rules"] if v["verdict"] == "not checkable"]
        rep = [v["id"] for v in o["rules"] if v["verdict"] == "reported"]
        line = (f"{'PASS' if o['passed'] else 'FAIL'}: {len(o['rules'])} declared rules, {o['n_must_failed']} must-hold rules failed, "
                f"reported {', '.join(rep) or 'none'}, not checkable {', '.join(nc) or 'none'}")
        if ds == "casia_fasd":
            line += (f"; official fitting and early stopping disjoint by key, hash, subject and video; pooled: {po.get('eval_images_in_pooled_fit_or_validation')} "
                     f"of 300 evaluation images in the pooled fitting or early stopping, {po.get('subjects_on_both_sides')} of {po.get('subjects_total')} subjects and "
                     f"{po.get('videos_on_both_sides')} of {po.get('videos_total')} videos on both pooled sides (declared counts)")
        else:
            line += (f"; one exact duplicate pair (declared); pooled: {po.get('eval_images_in_pooled_fit_or_validation')} of 300 evaluation images in the "
                     "pooled fitting or early stopping (declared count)")
        return line + f" (`{ds}_overlap_report_public.md`)"

    def ck_line():
        out = []
        for ds in DATASETS:
            for n, c in ((summ.get(ds) or {}).get("checkpoints") or {}).items():
                cc = c["card_counts"]
                out.append(f"{n} `{c['weights_sha256']}` (verified: {yes_no(c['weights_sha256_verified'])}; card {cc['n_train']:,}/{cc['n_val']:,}/{cc['n_test']:,} "
                           f"equal to the manifests: {yes_no(c['counts_match_manifests'])})")
        return "; ".join(out) or "not produced"

    st_line = "; ".join(f"{ds}: {'PASS' if (st.get(ds) or {}).get('passed') else 'FAIL'}, {(st.get(ds) or {}).get('n_detected_as_expected')} of "
                        f"{(st.get(ds) or {}).get('n_faults')} faulty copies rejected by exactly the expected rules" for ds in DATASETS)
    der = []
    for ds in DATASETS:
        d = (((summ.get(ds) or {}).get("reconstruction") or {}).get("training_cache") or {}).get("derived_frames")
        if d:
            der.append(f"{ds}: {d['train']['n_frames']:,} + {d['test']['n_frames']:,} frames, {d['train']['duplicated_values']} + {d['test']['duplicated_values']} "
                       f"duplicated values, {d['train_test_shared']} shared between train and test, {d['fit_vs_validation_shared']} between fitting and early "
                       f"stopping, {d['pooled_trainval_vs_test_shared']} between the pooled sides")
    rc = []
    for ds in DATASETS:
        r = (summ.get(ds) or {}).get("reconstruction") or {}
        if ds == "casia_fasd" and r:
            u = r.get("held_out_uniqueness") or {}
            rc.append(f"CASIA-FASD official by subject (`GroupShuffleSplit`, seed 42); the card counts alone do not pin the held-out subjects "
                      f"({u.get('n_with_card_n_val')} of {u.get('n_combinations'):,} combinations give 316 images)")
        elif r:
            rc.append("CelebA-Spoof \"official\" per image (`GroupShuffleSplit` on file names, hypothesis 1; a stratified split gives the same counts, "
                      "which one was used is not verified)")
    rc.append("pooled: 80/20 stratified, then 10 % stratified early stopping; the current `scripts/train_cnn.py` would not reproduce these counts")
    ex = []
    for ds in DATASETS:
        e = ((agg.get(ds) or {}).get("recount") or {}).get("exclusions") or {}
        for k, v in e.items():
            ex.append(f"{ds} {k.replace('_', ' ')}: {v['n']:,} ({v['n_bona_fide']:,} bona fide, {v['n_attack']:,} attacks)")
    celeba_src = (summ.get("celeba_spoof") or {}).get("source") or {}
    if celeba_src:
        ex.append(f"celeba_spoof rows without image bytes: {celeba_src.get('rows_without_image')} (excluded from every manifest)")
    passed_all = all((summ.get(ds) or {}).get("passed") for ds in DATASETS) and all((ov.get(ds) or {}).get("passed") for ds in DATASETS)
    res = (("Every manifest rebuilt from the copies is identical to the SHA-256 declared in the rules; the reconstructed splits of the twelve "
            "checkpoints match the counts of their cards and the label and group arrays of the training caches (the original split lists were "
            "not recorded); every must-hold rule passes; the declared leaks of the pooled checkpoints and the declared exact duplicate are "
            "confirmed with their counts.") if passed_all else "**Some checks failed: see below.**")
    export_cmd = "python scripts/export_m2_rgb.py" + (f" --run-id {run_id}" if a.run_id else "") + (" --strict" if a.strict else "") \
        + (" --trial" if getattr(a, "trial", False) else "") \
        + (f" --uploaded-run {a.uploaded_run} --uploaded-on {a.uploaded_on} --uploaded-sums <SHA256SUMS of that folder>" if getattr(a, "uploaded_run", None) else "") \
        + (" --out <scratch folder>" if Path(a.out).resolve() != (ROOT / "export").resolve() else "") \
        + (" --m2 <scratch folder>" if m2.resolve() != (ROOT / "results" / "m2_rgb").resolve() else "")
    m2_rel = rel(m2)
    restricted_main = "; ".join(f"`{ds}/{f}` — `{sha256(m2 / ds / f)}`" for ds in DATASETS
                                for f in (f"{ds}_manifests_summary.json", "overlap_report.json", f"{ds}_m2_report.md") if (m2 / ds / f).is_file())
    return dict(run_id=run_id, run_date=f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}", date=datetime.now(ROME).strftime("%Y-%m-%d"), commit=commit,
                tree_line="",
                export_cmd=export_cmd, m2_dir=m2_rel if not Path(m2_rel).is_absolute() else "<scratch folder>",
                rules_sha_casia=sha256(DELIV / "casia_fasd_overlap_rules.json"), rules_sha_celeba=sha256(DELIV / "celeba_spoof_overlap_rules.json"),
                reference_line="", hardware_line=hardware_line(), checkpoint_line=ck_line(),
                casia_revision=((summ.get("casia_fasd") or {}).get("source") or {}).get("revision"),
                casia_sha=((summ.get("casia_fasd") or {}).get("source") or {}).get("archive_sha256"),
                celeba_revision=celeba_src.get("revision"), celeba_sha=celeba_src.get("file_sha256"),
                manifests_line="; ".join(man_line(ds) for ds in DATASETS), casia_counts=counts("casia_fasd"), celeba_counts=counts("celeba_spoof"),
                casia_overlap=overlap("casia_fasd"), celeba_overlap=overlap("celeba_spoof"), self_test_line=st_line,
                derived_line="; ".join(der) or "not checked", reconstruction_line="; ".join(rc) + ".", restricted_line=restricted_main or "not produced",
                n_public=counts_files[0], n_restricted=counts_files[1], exclusions_line="; ".join(ex) or "none",
                verification_line=logs_info["verification"], build_line=logs_info["build"], tests_line=logs_info["tests"],
                package_tests_line=logs_info["package"], nuaa_tests_line=logs_info["nuaa"], export_tests_line=logs_info["export"],
                not_produced="Not produced in this run: " + ("; ".join(f"`{w}` ({r})" for w, r in not_produced) if not_produced else "none") + ".",
                result_line=res, failed_line=("Failed checks of this export: " + "; ".join(failed) + ".") if failed else
                "No failed check in this export (an export with a failed check exits with code 1 and is not delivered).",
                reviewer_line=REVIEW_LINE, authorisation_note=AUTHORISATION_NOTE)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export del pacchetto M2, seconda tranche (CASIA-FASD e CelebA-Spoof)")
    ap.add_argument("--out", default=str(ROOT / "export"), help="cartella di uscita (default export/; per le prove una cartella di scratch)")
    ap.add_argument("--m2", default=str(ROOT / "results" / "m2_rgb"), help="cartella dei risultati rigenerati (default results/m2_rgb)")
    ap.add_argument("--run-id", default=None, help="run-id (default AAAAMMGG-M2-rgb-seed42-<commit>)")
    ap.add_argument("--strict", action="store_true", help="albero pulito obbligatorio e codice 1 se manca un artefatto")
    ap.add_argument("--dry-run", action="store_true", help="mostra ingressi, destinazioni ed elenchi senza scrivere nulla")
    ap.add_argument("--trial", action="store_true", help="prova esplicita da un albero con modifiche non registrate (dichiarata nel run report)")
    ap.add_argument("--uploaded-run", default=None, help="run-id della cartella riservata già caricata (dichiarazione dell'operatore)")
    ap.add_argument("--uploaded-on", default=None, help="data del caricamento, AAAA-MM-GG (con --uploaded-run)")
    ap.add_argument("--uploaded-sums", default=None, help="SHA256SUMS della cartella riservata caricata (sola lettura)")
    a = ap.parse_args()
    # i modelli della scansione legati alla macchina stanno in un file locale mai pubblicato: senza, niente export --strict
    if a.strict and (msg := local_patterns_problem()):
        print(f"ERROR: strict mode: {msg}", file=sys.stderr); return 1
    up = (a.uploaded_run, a.uploaded_on, a.uploaded_sums)
    if any(up) and not all(up):
        print("ERROR: --uploaded-run, --uploaded-on and --uploaded-sums go together", file=sys.stderr); return 1
    if a.uploaded_run and (not RUN_ID_RX.match(a.uploaded_run) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", a.uploaded_on)
                           or not Path(a.uploaded_sums).is_file()):
        print("ERROR: --uploaded-run must be an RGB run-id, --uploaded-on a date YYYY-MM-DD and --uploaded-sums an existing SHA256SUMS",
              file=sys.stderr)
        return 1
    out, m2 = Path(a.out).resolve(), Path(a.m2).resolve()
    for ro in READ_ONLY:                                 # per identità dei file, non per stringhe (path_guard.within)
        if (ROOT / ro).exists() and path_guard.within(m2, ROOT / ro):
            print(f"ERROR: --m2 {rel(m2)} is inside the read-only area {ro}", file=sys.stderr); return 1
    if msg := deletion_problem(m2, a.run_id or ""):
        print(f"ERROR: --m2 cannot be regenerated: {msg}", file=sys.stderr); return 1
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "n/d"
    run_id = a.run_id or f"{today()}-M2-rgb-seed42-{commit[:7]}"
    dirty = repo_dirty()                    # stesso comando di stato dei JSON (rgb_manifests.STATUS_ARGS)
    state_start = tree_outside(m2)          # fuori da --m2, che l'export cancella e rigenera (stato dei JSON)
    if status_failed(state_start):
        print("ERROR: git status failed: the repository state cannot be recorded", file=sys.stderr); return 1
    run_id = effective_run_id(run_id, dirty, a.trial)
    if not RUN_ID_RX.match(run_id):
        print(f"ERROR: invalid run-id {run_id!r} (expected YYYYMMDD-M2-rgb-...)", file=sys.stderr); return 1
    repo = out / "research" / PUBLIC_NAME
    d = out / "drive" / "02_Experiments" / "M2" / run_id
    for what, p in (("--m2", m2), ("public folder", repo), ("restricted run folder", d)):   # tutte le cartelle che l'export ricrea
        if msg := deletion_problem(p, run_id):
            print(f"ERROR: {what} cannot be written: {msg}", file=sys.stderr); return 1
    inputs = {"README template": DELIV / "README.md", "run report template": DELIV / "run-report.md.tmpl",
              **{f"document {n}": DELIV / n for n in PUBLIC_DOCS + RESTRICTED_DOCS}, **{f"code {c}": ROOT / c for c in CODE}}
    if a.dry_run:
        print("run-id:", run_id); print("commit:", commit); print("results folder (regenerated):", rel(m2))
        print("public folder:", rel(repo)); print("restricted folder:", rel(d))
        for k, p in inputs.items():
            print(f"  {'ok     ' if p.exists() else 'MISSING'} {k}")
        return 1 if any(not p.exists() for p in inputs.values()) else 0
    gate = tree_gate(dirty, a.trial)
    if gate:
        print(f"ERROR: {gate}", file=sys.stderr); return 1
    if a.strict and (msg := nuaa_package_problem(ROOT)):
        print(f"ERROR: strict mode: {msg}", file=sys.stderr); return 1
    if a.strict:
        r = subprocess.run(["git", *RM.STATUS_ARGS, "--", *STRICT_PATHS], cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0 or r.stdout.strip():
            print(f"strict mode: uncommitted changes in the exported code or documents:\n{r.stdout[:2000]}nothing exported, exit 1")
            return 1
    missing = [k for k, p in inputs.items() if not p.exists()]
    if missing:
        print(f"ERROR: required inputs missing: {missing}", file=sys.stderr); return 1
    failed, not_produced, logs, rcs = [], [], {}, {}

    # 1) rigenerazione e test
    print(f"1) regenerate {rel(m2)} at commit {commit[:7]}", flush=True)
    safe_rmtree(m2, run_id)
    steps = []
    for ds in DATASETS:
        man = m2 / ds / "manifests"
        steps += [(f"{ds}_build.log", ["scripts/rgb_manifests.py", "build", "--dataset", ds, "--out-root", m2]),
                  (f"{ds}_check_overlap.log", ["scripts/rgb_manifests.py", "check-overlap", "--dataset", ds, "--self-test", "--out-root", m2]),
                  (f"{ds}_recount.log", ["scripts/rgb_manifests.py", "recount", "--dataset", ds, "--manifests-dir", man,
                                         "--aggregate", m2 / ds / f"{ds}_manifests_aggregate.json"]),
                  (f"{ds}_reconstruct.log", ["scripts/rgb_manifests.py", "reconstruct", "--dataset", ds, "--manifests-dir", man]),
                  (f"{ds}_report.log", ["scripts/rgb_manifests.py", "report", "--dataset", ds, "--out-root", m2])]
    steps += [("check_rgb_manifests.txt", ["scripts/check_rgb_manifests.py", "--require-real", "--real-dir", m2]),
              ("check_nuaa_manifests.txt", ["scripts/check_nuaa_manifests.py"]),
              ("check_export_m2_rgb.txt", ["scripts/check_export_m2_rgb.py"]),
              ("check_rgb_mutations.txt", ["scripts/check_rgb_mutations.py"])]
    for name, cmd in steps:
        rc, txt = run(PY, *cmd)
        rcs[name] = (rc, txt)
        logs[name] = redact(txt.replace(str(m2), "<m2>"))
        if rc != 0:
            failed.append(f"{' '.join(str(c) for c in cmd[:3])} exited with code {rc} ({name})")
    t_rgb, s_rgb = tests_line(rcs["check_rgb_manifests.txt"][1], rcs["check_rgb_manifests.txt"][0])
    t_nuaa, _ = tests_line(rcs["check_nuaa_manifests.txt"][1], rcs["check_nuaa_manifests.txt"][0])
    t_exp, s_exp = tests_line(rcs["check_export_m2_rgb.txt"][1], rcs["check_export_m2_rgb.txt"][0])
    for what, skip in (("check_rgb_manifests.py", s_rgb), ("check_export_m2_rgb.py", s_exp)):
        failed += skip_problems(what, rcs[what.replace(".py", ".txt")][1], skip)
    if msg := tree_change_problem(state_start, tree_outside(m2)):
        failed.append(msg)
    json_dirty = bool(state_start[2])
    failed += json_state_problems(m2, commit, json_dirty, state_start[0])
    builds = []
    for ds in DATASETS:
        m = re.search(r"build checks passed \((\d+) checks\)", rcs[f"{ds}_build.log"][1])
        builds.append(f"{ds} {m.group(1)} checks passed" if m else f"{ds} FAILED (see {ds}_build.log)")

    def load(p: Path):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            not_produced.append((rel(p) if not Path(rel(p)).is_absolute() else p.name, "not written by rgb_manifests.py (see the logs)"))
            failed.append(f"{p.name} missing or unreadable")
            return None
    summ = {ds: load(m2 / ds / f"{ds}_manifests_summary.json") for ds in DATASETS}
    agg = {ds: load(m2 / ds / f"{ds}_manifests_aggregate.json") for ds in DATASETS}
    ov = {ds: load(m2 / ds / "overlap_report_public.json") for ds in DATASETS}
    stt = {ds: load(m2 / ds / "overlap_self_test.json") for ds in DATASETS}
    for ds in DATASETS:
        if summ[ds] is not None and (summ[ds].get("passed") is not True or summ[ds].get("commit") != commit):
            failed.append(f"{ds} summary: passed is not true or built at another commit")
        if ov[ds] is not None and ov[ds].get("passed") is not True:
            failed.append(f"{ds} overlap report: passed is not true")
        failed += artefact_problems(m2 / ds, ds, commit)

    # 2) parte pubblica
    print("2) public folder", flush=True)
    safe_rmtree(repo, run_id); safe_rmtree(d, run_id)
    repo.mkdir(parents=True); d.mkdir(parents=True)
    for n in PUBLIC_DOCS:
        shutil.copy2(DELIV / n, repo / n)
    for ds in DATASETS:
        for src, dst in PUBLIC_RESULTS.items():
            s, t = m2 / ds / src.format(ds=ds), repo / dst.format(ds=ds)
            if s.is_file():
                shutil.copy2(s, t)
            else:
                not_produced.append((t.name, f"{s.name} not written")); failed.append(f"public {t.name} missing")
    code_root = repo / "livenesslab"
    for item in copy_code(code_root):
        not_produced.append((f"livenesslab/{item}", "file not present")); failed.append(f"code {item} missing")
    failed += code_import_problems(code_root)
    readme = (DELIV / "README.md").read_text(encoding="utf-8")
    for ph in ("{commit}", "{run_id}", "{delivery_status}"):
        if ph not in readme:
            failed.append(f"README.md template: placeholder {ph} missing")
    status, st_problems = delivery_status(run_id, a.uploaded_run, a.uploaded_on, a.uploaded_sums, m2)
    failed += st_problems
    if dirty and a.trial:
        status = TRIAL_NOTE + "\n\n" + status
    (repo / "README.md").write_text(readme.replace("{commit}", commit[:7]).replace("{run_id}", run_id)
                                    .replace("{delivery_status}", status), encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="m2-rgb-package-") as tmp:
        if msg := deletion_problem(tmp, run_id):             # cartella di lavoro temporanea, cancellata all'uscita
            raise SystemExit(f"ERROR: refusing the temporary folder: {msg}")
        work = Path(tmp) / "livenesslab"
        shutil.copytree(code_root, work)
        env = {**os.environ, "LIVENESSLAB_DATA_DIR": str(Path(tmp) / "data"), "LIVENESSLAB_RESULTS_DIR": str(Path(tmp) / "results"),
               "LIVENESSLAB_WEIGHTS_DIR": str(Path(tmp) / "weights"), "PYTHONDONTWRITEBYTECODE": "1"}
        rc, txt = run(PY, "scripts/check_rgb_manifests.py", env=env, cwd=work)
        logs["package_tests.txt"] = redact(txt).replace(str(work), "<package>")
        pk_tests, _ = tests_line(txt, rc)
        if rc != 0:
            failed.append(f"offline tests from the public livenesslab/ copy exited with code {rc}")
        rc, txt = run(PY, "scripts/check_export_m2_rgb.py", env=env, cwd=work)
        logs["package_export_tests.txt"] = redact(txt).replace(str(work), "<package>")
        pk_tests += f"; export tests from the same copy: {tests_line(txt, rc)[0]}"
        if rc != 0:
            failed.append(f"export tests from the public livenesslab/ copy exited with code {rc}")

    # 3) parte riservata
    print("3) restricted run folder", flush=True)
    for ds in DATASETS:
        (d / ds).mkdir()
        if (m2 / ds / "manifests").is_dir():
            shutil.copytree(m2 / ds / "manifests", d / ds / "manifests")
        else:
            not_produced.append((f"{ds}/manifests/", "not written by build")); failed.append(f"{ds} manifests missing")
        for f in RESTRICTED_RESULTS:
            s = m2 / ds / f.format(ds=ds)
            if s.is_file():
                shutil.copy2(s, d / ds / s.name)
            else:
                not_produced.append((f"{ds}/{s.name}", "not written")); failed.append(f"restricted {ds}/{s.name} missing")
    for n in RESTRICTED_DOCS:
        (d / n).write_text(fill_run_id((DELIV / n).read_text(encoding="utf-8"), run_id), encoding="utf-8")
    (d / "logs").mkdir()
    for name, txt in logs.items():
        (d / "logs" / name).write_text(txt, encoding="utf-8")
    (d / "pip_freeze.txt").write_text(subprocess.run([PY, "-m", "pip", "freeze"], capture_output=True, text=True).stdout, encoding="utf-8")
    (d / "NOT_PRODUCED.md").write_text("# Artefacts not produced in this run\n\n" + ("Every expected artefact was produced.\n" if not not_produced else
                                       "".join(f"- `{w}`: not produced: {r}\n" for w, r in not_produced)), encoding="utf-8")
    n_restricted = sum(1 for p in d.rglob("*") if p.is_file()) + 3          # + run-report.md, README.md, ancora
    n_public = sum(1 for p in repo.rglob("*") if p.is_file()) + 1           # + run-report.md

    # 4) run report, SHA256SUMS, ancora
    ok = {ds: {k: rcs.get(f"{ds}_{k}.log", (1, ""))[0] == 0 for k in ("recount", "reconstruct")} for ds in DATASETS}
    info = {"verification": "Checks without images: " + "; ".join(f"{ds}: `recount` {'equal to the aggregate' if ok[ds]['recount'] else 'FAILED'}, "
                                                                f"`reconstruct` {'identical' if ok[ds]['reconstruct'] else 'FAILED'}" for ds in DATASETS) + ".",
            "build": "; ".join(builds), "tests": t_rgb, "package": pk_tests, "nuaa": t_nuaa,
            "export": t_exp + "; mutation check (`check_rgb_mutations.py`): " + (re.search(r"mutations killed: \d+/\d+", rcs["check_rgb_mutations.txt"][1]) or [
                "summary not found"])[0]}
    fields = report_fields(summ, ov, stt, agg, run_id, commit, a, m2, (n_public, n_restricted), info, failed, not_produced)
    ref_lines = []
    for f, published in REFERENCE_FILES.items():
        h = published_sha(published)
        same = h is not None and sha256(ROOT / f) == h
        ref_lines.append(f"`{f}` `{sha256(ROOT / f)[:16]}…` {'=' if same else 'NOT VERIFIED AGAINST'} `{published}`")
        if h is not None and not same:
            failed.append(f"{f} differs from the copy delivered in the NUAA package")
    fields["reference_line"] = "; ".join(ref_lines)
    fields["tree_line"] = tree_line(dirty, state_start[0], json_dirty)
    (repo / "run-report.md").write_text((DELIV / "run-report.md.tmpl").read_text(encoding="utf-8").format(**fields), encoding="utf-8")
    shutil.copy2(repo / "run-report.md", d / "run-report.md")
    pub_sha, n_listed, sp = write_sums(repo)
    failed += sp
    if n_listed != n_public:
        failed.append(f"public SHA256SUMS lists {n_listed} files, the run report says {n_public}")
    (d / PUBLIC_ANCHOR).write_text(f"{pub_sha}  SHA256SUMS\n", encoding="utf-8")
    files = sorted({p.name for p in d.iterdir()} | {"SHA256SUMS"})
    descr = {**DESCR, "logs": "logs of every step (machine paths replaced by <repo>, ~, <tmp> and <m2>)"}
    (d / "README.md").write_text(
        f"# M2 run folder {run_id} (second tranche, CASIA-FASD and CelebA-Spoof)\n\n"
        + "\n".join(f"- `{f}{'/' if (d / f).is_dir() else ''}`: {descr.get(f, '')}" for f in files if f != "README.md")
        + f"\n\nAnchor of the public package: the `SHA256SUMS` of `research/{PUBLIC_NAME}/` has SHA-256 `{pub_sha}` and lists "
          f"{n_listed} files (the same value is in `{PUBLIC_ANCHOR}`). The manifests are per-image records: they stay in "
          "this restricted folder (licence documents and authorisations: see the public README). No image and no weight file is included.\n", encoding="utf-8")
    _, n_restricted_listed, sp = write_sums(d)
    failed += sp
    if n_restricted_listed != n_restricted:
        failed.append(f"restricted SHA256SUMS lists {n_restricted_listed} files, the run report says {n_restricted}")
    if sha256(repo / "SHA256SUMS") != pub_sha:
        failed.append("public SHA256SUMS changed after its anchor was written")

    # 5) scansione e completezza
    print("4) forbidden-content scan and completeness", flush=True)
    versions = env_versions()
    bad = [(f"research/{PUBLIC_NAME}/{f}", r) for f, r in scan(repo, versions)] + [(f"drive/.../{f}", r) for f, r in scan(d, versions)]
    ids = record_ids(d)
    if not ids["hashes"]:
        failed.append("no per-image manifest in the restricted folder: the public folder cannot be checked for per-image records")
    bad += [(f"research/{PUBLIC_NAME}/{f}", r) for f, r in public_record_problems(repo, ids)]
    failed += completeness_problems(repo) + completeness_problems(d)
    for f, r in bad:
        print(f"  ! forbidden content: {f}: {r}")
    print(f"public folder: {rel(repo)} ({n_listed} files + SHA256SUMS) · restricted folder: {rel(d)} ({n_restricted} files + SHA256SUMS) "
          f"· run-id {run_id} · not produced: {len(not_produced)}")
    for f in failed:
        print(f"  ! failed check: {f}")
    if bad or failed:
        return 1
    if a.strict and not_produced:
        return 1
    print("== export completed without problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
