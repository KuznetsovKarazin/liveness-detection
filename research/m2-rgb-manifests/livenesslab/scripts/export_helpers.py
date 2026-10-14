"""
Funzioni di servizio degli export M2 della seconda tranche (cartelle pubbliche e riservate): SHA-256, SHA256SUMS,
percorsi resi relativi nei log, versioni dell'ambiente, moduli importati, esecuzione dei comandi, riga dei test,
scansione del contenuto vietato, completezza, run-id nei documenti, riga dell'hardware.

Copia testuale delle funzioni di scripts/export_m1.py e scripts/export_m2.py (che restano invariati e non sono nel
pacchetto pubblico perché contengono elenchi riservati): scripts/check_export_m2_rgb.py verifica, nel repository di
sviluppo, che il codice di ogni funzione (a parte le docstring, riformulate qui in modo neutro) e FORBIDDEN_EXT siano
identici agli originali. Le regole del contenuto vietato vengono da scripts/package_rules.py, come in export_m2.py.
"""
import ast
import re
import subprocess
import sys
from pathlib import Path

from package_rules import FORBIDDEN_TEXT, IP_ALLOW, IPV4, MARKERS  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# estensioni vietate (stesso insieme di export_m1.py)
FORBIDDEN_EXT = {'.bin', '.bmp', '.ckpt', '.gif', '.gz', '.h5', '.hdf5', '.heic', '.joblib', '.jpeg', '.jpg', '.keras', '.mlmodel', '.mlpackage', '.npy', '.npz', '.onnx', '.pb', '.pgm', '.pickle', '.pkl', '.png', '.ppm', '.pt', '.pth', '.safetensors', '.tar', '.task', '.tflite', '.tif', '.tiff', '.webp', '.zip'}


def sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(*cmd, env=None, cwd=None):
    """Esegue un comando; restituisce (codice, stdout + stderr). Nessuna eccezione: l'esito decide il chiamante."""
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd or ROOT, capture_output=True, text=True, env=env)
    return r.returncode, r.stdout + r.stderr


def tests_line(txt: str, rc: int) -> tuple:
    """Riga di riepilogo di una suite unittest del repository e numero di test saltati (None se il riepilogo manca)."""
    m = re.search(r"^tests run: (\d+), failures: (\d+), errors: (\d+), skipped: (\d+)", txt, re.M)
    if not m:
        return f"summary not found, exit code {rc}", None
    return f"{m.group(1)} tests, {m.group(2)} failures, {m.group(3)} errors, {m.group(4)} skipped, exit code {rc}", int(m.group(4))


def scan(folder: Path, versions: set) -> list:
    """Contenuto vietato in una cartella dell'export: [(file, motivo)]. Estensioni di immagini e pesi; nei file di testo
    le regole di package_rules.py (percorsi assoluti, nomi dell'operatore o di host, credenziali, riferimenti a strumenti),
    indirizzi IPv4 non locali e diversi da una versione dell'ambiente, segni di lavoro non finito nei .md."""
    bad = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        r = p.relative_to(folder).as_posix()
        if p.is_symlink():
            bad.append((r, "symbolic link")); continue
        if p.suffix.lower() in FORBIDDEN_EXT:
            bad.append((r, f"forbidden file type {p.suffix}")); continue
        try:
            txt = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            bad.append((r, "binary file")); continue
        for label, rx in FORBIDDEN_TEXT:
            if m := rx.search(txt):
                bad.append((r, f"{label}: {m.group(0)[:40]!r}"))
        for ip in (m.group(0) for m in IPV4.finditer(txt)):
            if ip not in IP_ALLOW and ip not in versions:
                bad.append((r, f"IP address: {ip}")); break
        if p.suffix.lower() == ".md":
            for i, line in enumerate(txt.splitlines(), 1):
                if m := MARKERS.search(line):
                    bad.append((r, f"unfinished-work marker {m.group(0)!r} at line {i}")); break
    return bad


def completeness_problems(folder: Path) -> list:
    """Come il comando di completezza del README: ogni file elencato in SHA256SUMS esiste con lo stesso SHA-256, nessun
    file in più o in meno, nessun collegamento simbolico."""
    sums = folder / "SHA256SUMS"
    if not sums.is_file():
        return [f"{folder.name}: SHA256SUMS missing"]
    listed = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not m:
            return [f"{folder.name}: malformed SHA256SUMS line {line[:60]!r}"]
        listed[m.group(2)] = m.group(1)
    present = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() or p.is_symlink()} - {"SHA256SUMS"}
    problems = []
    links = [p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_symlink()]
    if links:
        problems.append(f"{folder.name}: symbolic links {links[:3]}")
    if set(listed) != present:
        problems.append(f"{folder.name}: listed files differ from present files (missing {sorted(set(listed) - present)[:3]}, "
                        f"unlisted {sorted(present - set(listed))[:3]})")
    wrong = [f for f, h in listed.items() if (folder / f).is_file() and sha256(folder / f) != h]
    if wrong:
        problems.append(f"{folder.name}: SHA-256 differs for {wrong[:3]}")
    return problems


def fill_run_id(text: str, run_id: str) -> str:
    """Sostituisce il segnaposto {run_id} di un documento sorgente; errore se il segnaposto manca o resta."""
    if "{run_id}" not in text:
        raise ValueError("placeholder {run_id} missing")
    out = text.replace("{run_id}", run_id)
    if "{run_id}" in out or "<run-id>" in out:
        raise ValueError("run-id placeholder left in the document")
    return out


def hardware_line() -> str:
    import platform
    vers = []
    for mod in ("numpy", "sklearn"):
        try:
            vers.append(f"{'scikit-learn' if mod == 'sklearn' else mod} {__import__(mod).__version__}")
        except Exception:  # noqa: BLE001
            vers.append(f"{mod} not available")
    osname = f"macOS {platform.mac_ver()[0]}" if platform.mac_ver()[0] else platform.system()
    return (f"development machine ({osname}, {platform.machine()}), Python {platform.python_version()}, {', '.join(vers)}. "
            "File listing and hashing only: no inference, no training, no model loaded.")


def write_sums(folder: Path) -> tuple:
    """SHA256SUMS della cartella (percorsi relativi, anche nelle sottocartelle, formato di `shasum -a 256`), scritto per
    ultimo. Restituisce (SHA-256 del file SHA256SUMS, numero di file elencati, problemi)."""
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.relative_to(folder).as_posix() != "SHA256SUMS")
    bad = [p.relative_to(folder).as_posix() for p in files if re.search(r"\s", p.relative_to(folder).as_posix())]
    (folder / "SHA256SUMS").write_text("".join(f"{sha256(f)}  {f.relative_to(folder).as_posix()}\n" for f in files), encoding="utf-8")
    return sha256(folder / "SHA256SUMS"), len(files), [f"{folder.name}: file name with whitespace breaks the completeness command: {b}" for b in bad]


def redact(txt: str) -> str:
    """Percorsi della macchina resi relativi: repository -> <repo>, home -> ~, cartelle temporanee -> <tmp>."""
    txt = txt.replace(str(ROOT), "<repo>").replace(str(Path.home()), "~")
    return re.sub(r"(?:/private)?/(?:tmp|var/folders)/[^\s\"\)]+", "<tmp>", txt)


def env_versions() -> set:
    """Versioni fissate nei file dell'ambiente (es. 4.11.0.86): hanno la forma di un IPv4 ma non lo sono."""
    vs = set()
    for f in ("requirements.txt", "requirements-windows.txt"):
        p = ROOT / f
        if p.exists():
            vs |= set(re.findall(r"==\s*([\d.]+)", p.read_text(encoding="utf-8")))
    vs |= set(re.findall(r"==\s*([\d.]+)", subprocess.run([PY, "-m", "pip", "freeze"], capture_output=True, text=True).stdout))
    return vs


def local_imports(py: Path) -> set:
    """Moduli di scripts/ importati da un file (import x / from x import y con x = file di scripts/)."""
    names = {p.stem for p in (ROOT / "scripts").glob("*.py")}
    tree = ast.parse(py.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return {n for n in found if n in names}
