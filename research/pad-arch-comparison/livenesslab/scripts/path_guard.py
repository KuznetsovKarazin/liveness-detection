"""
Protezione delle cartelle che un export cancella o riscrive (comune agli export di C1, M1 e M2).

Il confronto è per identità dei file (os.path.samefile), non per stringhe: su un filesystem che non distingue le
maiuscole (per esempio quello predefinito di macOS) due grafie diverse della stessa cartella sono la stessa cartella, e
un confronto fra stringhe le tratterebbe come cartelle diverse. Per un percorso che non esiste ancora si usa il suo
antenato esistente più vicino, con il resto del percorso confrontato senza distinzione di maiuscole.

Una cartella è rifiutata se è la radice del filesystem, la home o un suo antenato, la radice del repository o un suo
antenato, oppure uguale, antenata o interna a una cartella protetta; dentro il repository sono ammesse solo le cartelle
dell'export che chiama (allowed); fuori dal repository le altre cartelle (per esempio di scratch).
"""
import os
import shutil
from pathlib import Path

# cartelle del repository che nessun export cancella o riscrive (ogni export toglie dall'elenco solo le proprie)
BASE_PROTECTED = ["data", "models", "results", "results/c1", "results/m1", "results/m2", "results/m2_rgb", "results/eval",
                  "docs", "scripts", "src", "delivery", "config", ".git", "export", "export/research", "export/drive"]


def chain(p) -> list:
    """Il percorso risolto e i suoi antenati, solo quelli che esistono (dal più vicino alla radice)."""
    p = Path(p).resolve()
    return [q for q in (p, *p.parents) if q.exists()]


def same(a, b) -> bool:
    """Stessa cartella sul disco (stesso dispositivo e inode), qualunque sia la grafia."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def contains(p, q) -> bool:
    """p è q o un suo antenato (q deve esistere)."""
    return any(same(p, x) for x in chain(q))


def within(p, q) -> bool:
    """p è q o sta dentro q. Se q non esiste, si confronta il suo antenato esistente più vicino per identità e il resto
    del percorso senza distinzione di maiuscole."""
    p, q = Path(p).resolve(), Path(q).resolve()
    if q.exists():
        return any(same(q, x) for x in chain(p))
    base = chain(q)[0]
    tail = [s.casefold() for s in q.relative_to(base).parts]
    for x in (p, *p.parents):
        if x.exists() and same(x, base):
            rest = [s.casefold() for s in p.relative_to(x).parts]
            return rest[:len(tail)] == tail
    return False


def deletion_problem(path, root, protected, allowed, home=None) -> str:
    """Motivo per cui `path` non si può cancellare o riscrivere ("" se si può). `protected` sono percorsi relativi a
    `root`; `allowed` le cartelle (anche non ancora esistenti) che l'export può ricreare dentro il repository."""
    p = Path(path).resolve()
    root = Path(root).resolve()
    home = Path(home if home is not None else Path.home()).resolve()
    if contains(p, Path(p.anchor)):
        return f"{p} is the filesystem root"
    if contains(p, home):
        return f"{p} is the home folder or one of its ancestors"
    if contains(p, root):
        return f"{p} is the repository root or one of its ancestors"
    prot = [root / x for x in protected if (root / x).exists()]
    for q in prot:
        if contains(p, q):
            return f"{p} contains the protected folder {q.relative_to(root)}"
    if not within(p, root):
        return ""
    if any(within(p, a) for a in allowed):
        return ""
    inside = [q for q in prot if within(p, q)]
    where = f"the protected folder {inside[-1].relative_to(root)}" if inside else "the repository"
    return f"{p} is inside {where} and is not a folder of this export"


def safe_rmtree(path, root, protected, allowed, home=None, ignore_errors: bool = True) -> None:
    """Cancella una cartella solo dopo deletion_problem, subito prima della cancellazione."""
    msg = deletion_problem(path, root, protected, allowed, home)
    if msg:
        raise SystemExit(f"ERROR: refusing to delete: {msg}")
    shutil.rmtree(path, ignore_errors=ignore_errors)
