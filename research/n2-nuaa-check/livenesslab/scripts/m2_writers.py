"""
Scritture sicure delle uscite della scheda M2 (manutenzione del 22/10/2026, richiesta del coordinatore dopo il merge
della PR #8), comuni a scripts/rgb_manifests.py e scripts/nuaa_manifests.py, e dichiarazione delle funzioni toccate
dalla manutenzione rispetto alle copie dei pacchetti accettati.

Difetto corretto: il write_all delle copie accettate scriveva ogni file in un temporaneo a nome fisso (<nome>.tmp) con
write_text, che segue un collegamento simbolico (e scrive nello stesso inode di un collegamento fisso): un temporaneo
preparato prima come collegamento faceva sovrascrivere un file esterno. Qui:
  - write_all crea ogni temporaneo con tempfile.mkstemp nella cartella di uscita (nome nuovo, O_CREAT | O_EXCL, nessun
    collegamento seguito), poi lo rinomina con os.replace, che sostituisce il nome e mai il file a cui un collegamento
    (simbolico o fisso) puntava; un'uscita che è un collegamento simbolico o non è un file regolare è rifiutata; su
    qualunque eccezione non resta nessun file del gruppo né alcun temporaneo, e un collegamento si toglie come
    collegamento, mai il suo bersaglio;
  - invalidate toglie le uscite precedenti e i loro temporanei (nome fisso delle copie accettate e nomi univoci di
    write_all) senza mai seguire un collegamento e rifiuta una cartella con il nome di un'uscita;
  - output_problem valida la cartella di uscita prima di qualunque modifica: non è un collegamento né un file, non sta
    nell'albero di lavoro che contiene i pacchetti accettati (outer_roots, come in c2_pilot.py), non viola path_guard
    (radice, home, repository, cartelle protette; dentro il repository solo le cartelle del comando), non è, non
    contiene e non sta dentro le cartelle dei dati, dei pesi e dei risultati consegnati (anche se spostate con le
    variabili d'ambiente), non è e non sta dentro una cartella d'ingresso, e nessun ingresso è uno dei file che il
    comando scrive o toglie (né sta dentro una cartella che il comando riscrive).

La cartella d'ingresso dei manifest può stare dentro la cartella di uscita (disposizione predefinita <out>/manifests,
usata da build -> check-overlap -> report): i comandi scrivono e tolgono solo i nomi dichiarati direttamente nella
cartella di uscita (e, per build, <out>/manifests), e un ingresso con uno di quei percorsi è rifiutato.

Libreria standard più scripts/path_guard.py; nessuna dipendenza esterna.
"""
import ast
import functools
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import path_guard as pg

_UMASK = os.umask(0o022)          # solo lettura della maschera corrente, ripristinata subito
os.umask(_UMASK)


def _fail(msg: str):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ============================================================================ alberi con i pacchetti accettati

@functools.lru_cache(maxsize=None)
def _outer_roots(root: str) -> tuple:
    """Alberi che contengono il codice senza esserne la radice (stessa regola di c2_pilot.py): l'albero di lavoro git
    che contiene root (git rev-parse --show-toplevel, se diverso da root) e ogni antenato X di root con root dentro
    X/research (disposizione del repository del gruppo, anche in una copia senza git). Lì stanno i pacchetti accettati:
    nessuna uscita è ammessa al loro interno. Nel repository di sviluppo, dove il codice è alla radice, nessun albero."""
    root = Path(root).resolve()
    out = []
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root, capture_output=True, text=True, timeout=30)
        top = Path(r.stdout.strip()).resolve() if r.returncode == 0 and r.stdout.strip() else None
        if top is not None and not pg.same(top, root):
            out.append(top)
    except (OSError, subprocess.SubprocessError):
        pass
    for x in root.parents:
        if (x / "research").is_dir() and any(pg.same(x / "research", y) for y in root.parents) and not any(pg.same(x, o) for o in out):
            out.append(x)
    return tuple(out)


def outer_roots(root) -> tuple:
    return _outer_roots(str(Path(root)))


# ============================================================================ validazione della cartella di uscita

def output_problem(out, root, protected, allowed, extra=(), inputs=None, outputs=(), dirs=()) -> str:
    """Motivo per cui `out` non può ricevere le uscite di un comando ("" se può), controllato prima di qualunque
    scrittura o cancellazione. root, protected, allowed: come path_guard.deletion_problem (protected relativi a root,
    allowed le cartelle del comando dentro il repository); extra: cartelle protette con percorso assoluto (dati, pesi,
    risultati consegnati, anche spostati con le variabili d'ambiente), rifiutate in entrambe le direzioni; inputs:
    etichetta -> percorso degli ingressi del comando; outputs: nomi dei file che il comando scrive o toglie in `out`;
    dirs: sottocartelle che il comando riscrive (per build, <out>/manifests)."""
    out = Path(out)
    for w in [out, *map(Path, dirs)]:
        if w.is_symlink():
            return f"{w.name} is a symbolic link: output folders must be real folders"
        if w.exists() and not w.is_dir():
            return f"{w.name} exists and is not a folder"
    for g in outer_roots(root):
        if pg.within(out, g) or pg.contains(out, g):
            return f"{out} is inside, or contains, the work tree {g.name} with the accepted packages: outputs go outside it"
    msg = pg.deletion_problem(out, root, protected, allowed)
    if msg:
        return msg
    for q in extra:
        if pg.within(out, q) or (Path(q).exists() and pg.contains(out, q)):
            return f"{out} is, contains or is inside a protected folder ({Path(q).name})"
    for label, p in (inputs or {}).items():
        p = Path(p)
        if pg.within(out, p):
            return f"the output folder is {label} {p.name} or inside it: inputs and outputs must be separate"
        # anche ogni voce di una cartella d'ingresso (per esempio un manifest che è un collegamento a un'uscita)
        for what, q in [(label, p)] + [(f"{label} entry", e) for e in entries_of(p)]:
            if hit := touched_by(q, out, outputs, dirs):
                return f"{what} {q.name} is, or is inside, a path this command removes or rewrites ({hit}): inputs and outputs must be separate"
    # verifica preliminare (revisione del docente del 23/10): ogni percorso che la pulizia toccherà deve essere un file
    # regolare o un collegamento (tolto come collegamento), non immutabile né «solo aggiunta», in una cartella in cui si
    # può scrivere; altrimenti rifiuto prima di togliere qualsiasi file
    for d in [out, *map(Path, dirs)]:
        if d.is_dir() and not os.access(d, os.W_OK | os.X_OK):
            return f"the folder {d.name} does not accept writes: its outputs could not be removed; refused before any change, nothing removed"
    for q in owned_paths(out, outputs) + [e for d in map(Path, dirs) if d.is_dir() for e in sorted(d.iterdir())]:
        st = os.lstat(q)
        if not (stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode)):
            mode = st.st_mode
            kind = "a folder" if stat.S_ISDIR(mode) else "a FIFO" if stat.S_ISFIFO(mode) else "a socket" if stat.S_ISSOCK(mode) else "not a regular file"
            return f"{q.name} in the output folder is {kind}, not a removable output: refused before any change, nothing removed"
        if getattr(st, "st_flags", 0) & LOCK_FLAGS or (os.name == "nt" and not os.access(q, os.W_OK)):
            return f"{q.name} in the output folder is immutable, append-only or read-only: it could not be removed; refused before any change, nothing removed"
    return ""


# indicatori che impediscono di togliere un file (BSD/macOS: chflags uchg/schg, uappnd/sappnd)
LOCK_FLAGS = getattr(stat, "UF_IMMUTABLE", 0) | getattr(stat, "SF_IMMUTABLE", 0) | getattr(stat, "UF_APPEND", 0) | getattr(stat, "SF_APPEND", 0)


def entries_of(p) -> list:
    """Voci (non ricorsive) di una cartella d'ingresso; [] per un file o un percorso illeggibile."""
    try:
        return sorted(Path(p).iterdir()) if Path(p).is_dir() else []
    except OSError:
        return []


def _ids(p) -> set:
    """Identità (dispositivo, inode) della voce `p` e del file a cui punta, se esistono."""
    out = set()
    for f in (os.lstat, os.stat):
        try:
            st = f(p)
            out.add((st.st_dev, st.st_ino))
        except OSError:
            pass
    return out


def owned_name(entry: str, n: str) -> bool:
    """Unica regola dei nomi che la pulizia di un'uscita `n` tocca (revisione del docente del 23/10): l'uscita, il
    temporaneo a nome fisso delle copie accettate (<n>.tmp) e i temporanei univoci di write_all (.<n>.<casuale>.tmp)."""
    return entry == n or entry == n + ".tmp" or (entry.startswith(f".{n}.") and entry.endswith(".tmp"))


def owned_paths(out, names) -> list:
    """Voci esistenti di `out` che invalidate toglierebbe per le uscite `names` (stessa regola di owned_name)."""
    try:
        entries = sorted(os.listdir(out))
    except OSError:
        return []
    return [Path(out) / e for e in entries if any(owned_name(e, n) for n in names)]


def touched_by(p, out, outputs, dirs=()) -> str:
    """Percorso (nome) che il comando toglie o riscrive e che coincide con l'ingresso `p` ("" se nessuno): `p` (come
    scritto e risolto) è una voce di `out`, per identità della cartella, con un nome della regola owned_name, oppure è,
    o sta dentro, una cartella che il comando riscrive."""
    out = Path(out)
    # per identità: le voci reali che invalidate toglierebbe (os.listdir), confrontate con l'ingresso per dispositivo e
    # inode, così un'altra grafia (filesystem che non distingue le maiuscole) o un alias sono riconosciuti
    removed = {}
    for q in owned_paths(out, outputs) if out.is_dir() else []:
        try:
            st = os.lstat(q)
            removed[(st.st_dev, st.st_ino)] = q.name
        except OSError:
            pass
    for cand in {Path(os.path.abspath(p)), Path(p).resolve()}:
        for i in _ids(cand):
            if i in removed:
                return f"{removed[i]} (the same file)"
        # per nome, senza distinzione di maiuscole, anche per le voci che non esistono ancora
        if out.exists() and pg.same(cand.parent, out):
            for n in outputs:
                if owned_name(cand.name.casefold(), n.casefold()):
                    return n if cand.name == n else f"{cand.name}, a temporary of {n}"
        for d in map(Path, dirs):
            if pg.within(cand, d):
                return d.name
    return ""


def foreign_outputs_problem(out, names, what: str) -> str:
    """Una cartella di uscita che contiene le uscite di un'altra tranche è rifiutata prima di toccare qualsiasi file."""
    found = sorted(n for n in names if os.path.lexists(Path(out) / n))
    if found:
        return f"output folder {Path(out).name} holds the outputs of {what} ({', '.join(found)}): refused, nothing removed"
    return ""


# ============================================================================ invalidazione e scrittura atomica

def temp_names(out, n: str) -> list:
    """Temporanei di un'uscita lasciati da un'esecuzione interrotta: il nome fisso delle copie accettate (<nome>.tmp) e
    i nomi univoci di write_all (.<nome>.<casuale>.tmp); stessa regola di owned_name."""
    return [q for q in owned_paths(out, [n]) if q.name != n]


def invalidate(out, names) -> list:
    """Toglie le uscite di un'esecuzione precedente e i loro temporanei prima di qualunque controllo: nessun artefatto
    vecchio (per esempio un PASS) resta accanto a un'esecuzione che fallisce. Un collegamento (anche senza bersaglio) si
    toglie come collegamento, il suo bersaglio mai; una cartella con il nome di un'uscita è rifiutata. Restituisce i
    nomi tolti."""
    removed = []
    for n in names:
        for q in owned_paths(out, [n]):                 # stessa regola della verifica degli ingressi (owned_name)
            if q.is_symlink():
                q.unlink()
                removed.append(q.name)
            elif q.is_dir():
                _fail(f"{q.name} in the output folder is a folder, not an output file: refused")
            elif q.exists():
                q.unlink()
                removed.append(q.name)
    return removed


CLEANUP_DOC = {"stage": "invalidation-failed", "note": "this output could not be removed after a failure: no verdict is valid"}


def invalidate_all(out, names, failed_doc: dict = None) -> list:
    """invalidate delle uscite all'inizio di un comando (revisione del docente del 23/10). Se si ferma a metà (anche con
    SystemExit, per esempio un oggetto non regolare comparso dopo la verifica preliminare), la pulizia è completata un
    nome alla volta (invalidate_each: le altre uscite tolte, i JSON che non si tolgono sovrascritti con un FAIL, i file
    rimasti nominati) e poi il comando si ferma con un rifiuto esplicito: nessun PASS resta accanto a un'invalidazione
    parziale. Un'interruzione da tastiera è rilanciata dopo la stessa pulizia."""
    try:
        return invalidate(out, names)
    except BaseException as e:
        left = invalidate_each(out, names, None, failed_doc or CLEANUP_DOC)
        if isinstance(e, KeyboardInterrupt):
            raise
        _fail(f"the invalidation of the previous outputs stopped ({type(e).__name__}): " + (
            "the other outputs were removed or marked FAIL" if not left else
            f"outputs that could be neither removed nor marked FAIL: {', '.join(left)}; the others were removed or marked FAIL"))


def invalidate_each(out, names, invalidate_fn, failed_doc: dict) -> list:
    """Pulizia dopo un errore (self_test_failure, unexpected_failure), un nome alla volta con la invalidate del modulo:
    se un'uscita non si può togliere (per esempio EACCES), un rapporto JSON è sovrascritto in modo atomico con un
    documento FAIL (failed_doc, senza testo d'errore); se nemmeno questo riesce, una riga esplicita «could not invalidate
    <nome>» la nomina. Restituisce i nomi rimasti; il comando esce comunque con codice diverso da 0."""
    left = []
    for n in names:
        try:
            (invalidate_fn or invalidate)(out, [n])     # None: invalidate di questo modulo, letta al momento della chiamata
            continue
        except (Exception, SystemExit) as e:  # noqa: BLE001  (doppio guasto: mai in silenzio)
            why = f"{type(e).__name__}: {e}"
        if n.endswith(".json"):
            try:
                write_all(out, {n: json.dumps({**failed_doc, "passed": False}, indent=1) + "\n"})
                print(f"ERROR: could not invalidate {n} ({why}): overwritten with a FAIL document", file=sys.stderr, flush=True)
                continue
            except (Exception, SystemExit):  # noqa: BLE001
                pass
        print(f"ERROR: could not invalidate {n} ({why}): left behind", file=sys.stderr, flush=True)
        left.append(n)
    return left


def write_all(out, files: dict, newline=None) -> None:
    """Scrive un gruppo di file (nome -> testo) solo alla fine: prima tutti i temporanei, creati in modo sicuro
    (tempfile.mkstemp nella cartella di uscita: nome nuovo, O_CREAT | O_EXCL, nessun collegamento seguito), poi le
    rinomine (os.replace sostituisce il nome, mai il file a cui un collegamento puntava). Un'uscita che è un collegamento
    simbolico o non è un file regolare è rifiutata prima di scrivere. Se la scrittura fallisce (qualunque eccezione) non
    resta nessun file del gruppo né alcun temporaneo, e nessun file esterno è toccato. newline come in open(): None per
    i testi (come write_text), "" per i CSV (come open(..., newline=""))."""
    out = Path(out)
    if out.is_symlink():
        _fail(f"{out.name} is a symbolic link: output folders must be real folders")
    out.mkdir(parents=True, exist_ok=True)
    tmps = {}
    try:
        for n in files:
            q = out / n
            if q.is_symlink() or (q.exists() and not q.is_file()):
                _fail(f"{n} in the output folder is a link or not a regular file: refused")
        for n, text in files.items():
            fd, tp = tempfile.mkstemp(prefix=f".{n}.", suffix=".tmp", dir=out)
            tmps[n] = Path(tp)
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o666 & ~_UMASK)            # stessi permessi di write_text
            with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as f:
                f.write(text)
        for n, tp in tmps.items():
            os.replace(tp, out / n)
    except BaseException:
        for tp in tmps.values():
            tp.unlink(missing_ok=True)
        for n in files:
            q = out / n
            if q.is_symlink() or q.is_file():           # il collegamento si toglie, il suo bersaglio mai
                q.unlink()
        raise


# ============================================================================ dichiarazione della manutenzione

# copie accettate (pacchetti di research/ del repository del gruppo) e funzioni o costanti di primo livello toccate
# dalla manutenzione: fuori da queste e dalle istruzioni aggiunte, il file corrente deve restare identico alla copia
# accettata (confronto dei nodi dell'AST); `current` è lo SHA-256 esatto del file corretto, da aggiornare a ogni modifica
ACCEPTED_COMMIT = "a5fb3ce"         # commit del repository di sviluppo con i file identici alle copie accettate
MAINT_LABEL = "= accepted copy except the writer maintenance of 22 October 2026"
MAINTENANCE = {
    "scripts/rgb_manifests.py": {
        "package": "m2-rgb-manifests",
        "accepted": "61e8d1a668b21488134d444cf3df31995e4b06dc985e3708adb4444089a4d68a",
        "current": "4036a3fa7833d845952564c4a5440d9061649ffebca7bcdfaf354ec4f6f6198e",
        "changed": ("invalidate", "write_all", "write_manifest", "write_rows", "outputs_problem", "cmd_build",
                    "VALIDATED", "self_test_faults", "run_self_test", "cmd_check_overlap", "self_test_failure", "cmd_report",
                    "main", "unexpected_failure"),
        "added": ("import m2_writers as mw",),
    },
    "scripts/nuaa_manifests.py": {
        "package": "m2-nuaa-manifests",
        "accepted": "bc1ce7b1c50eadbcbfb2c2817c5d846b6f9224d5848f31e33ef6923bb58450e5",
        "current": "a719ce2ed83479ab07883398d21eb1d869a4d4be273dda04b8cf34f8c99cfa44",
        "changed": ("invalidate", "write_all", "write_manifest", "write_rows", "outputs_problem", "cmd_build",
                    "VALIDATED", "run_self_test", "cmd_check_overlap", "overlap_run", "self_test_failure", "cmd_report",
                    "main", "unexpected_failure", "RGB_OUTPUTS"),
        "added": ("import m2_writers as mw",),
    },
    "scripts/export_m2_rgb.py": {
        "package": "m2-rgb-manifests",
        "accepted": "fa5b1eea8f527ceedb0d17e8f05da92f14ed8557866e7487d8f1b2bad5c4ff62",
        "current": "862738857ffd74ac4bea4cd44a63b0b8a128333f90ca0dd0fe354b95ccba60f5",
        "changed": ("CODE", "deletion_problem", "export_paths_problem", "reference_state", "reference_report", "main"),
        "added": ("import m2_writers as mw",),
    },
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def outside_nodes(text: str, changed, added=()) -> list:
    """Nodi di primo livello (ast.dump, in ordine) esclusi le funzioni e le costanti dichiarate (per nome) e le istruzioni
    aggiunte dichiarate (confrontate con ast.unparse)."""
    out = []
    for n in ast.parse(text).body:
        if isinstance(n, ast.FunctionDef) and n.name in changed:
            continue
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in changed:
            continue
        if ast.unparse(n) in added:
            continue
        out.append(ast.dump(n))
    return out


def accepted_text(rel: str, root) -> str:
    """Testo della copia accettata di `rel`, solo se il suo SHA-256 è quello dichiarato: dal pacchetto accanto (research/
    del repository del gruppo), da export/research/ del repository di sviluppo o dal commit ACCEPTED_COMMIT; None se
    nessuno è disponibile."""
    m, root = MAINTENANCE[rel], Path(root)
    cands = []
    for base in (root.parent.parent / m["package"], root / "export" / "research" / m["package"]):
        p = base / "livenesslab" / rel
        if p.is_file():
            cands.append(p.read_bytes())
    try:
        r = subprocess.run(["git", "show", f"{ACCEPTED_COMMIT}:{rel}"], cwd=root, capture_output=True, timeout=30)
        if r.returncode == 0:
            cands.append(r.stdout)
    except (OSError, subprocess.SubprocessError):
        pass
    return next((c.decode("utf-8") for c in cands if sha256_bytes(c) == m["accepted"]), None)


def maintenance_state(rel: str, data: bytes, accepted: str) -> str:
    """"=" se i byte sono quelli della copia accettata; MAINT_LABEL solo se i byte hanno esattamente lo SHA-256
    dichiarato del file corretto e, fuori dalle funzioni e dalle istruzioni dichiarate, l'AST coincide con quello della
    copia accettata verificata; "" in ogni altro caso (anche senza la copia accettata)."""
    m = MAINTENANCE[rel]
    h = sha256_bytes(data)
    if h == m["accepted"]:
        return "="
    if h != m["current"] or accepted is None or sha256_bytes(accepted.encode("utf-8")) != m["accepted"]:
        return ""
    try:
        same = outside_nodes(data.decode("utf-8"), m["changed"], m["added"]) == outside_nodes(accepted, m["changed"], m["added"])
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return ""
    return MAINT_LABEL if same else ""
