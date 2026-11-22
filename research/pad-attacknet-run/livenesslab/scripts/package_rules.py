"""
Costanti comuni ai pacchetti pubblici (M2 e successivi): regole del contenuto vietato e SHA-256 del manifest C1
consegnato (versione 1, congelata).

Copia dei valori di scripts/export_m1.py, che resta invariato: scripts/check_nuaa_manifests.py verifica che pattern e
flag di FORBIDDEN_TEXT e C1_MANIFEST_SHA256 siano identici a quelli di export_m1.py. Le parole vietate sono spezzate
nel sorgente, così questo file supera lo stesso controllo quando entra in un pacchetto pubblico.

I modelli legati alla macchina e agli strumenti (nomi dell'operatore o di host, i loro indirizzi, nomi di strumenti)
non stanno qui: si leggono all'import da config/local_scan_patterns.json del repository di sviluppo, file locale mai
pubblicato (in .gitignore e fuori da ogni export). Senza quel file (pacchetto pubblico) restano le regole generiche:
percorsi assoluti, credenziali, indirizzi IPv4 non locali, segni di lavoro non finito.
"""
import json
import os
import re
from pathlib import Path

# SHA-256 del manifest C1 (results/c1/nuaa_manifest.csv) come registrato alla consegna di C1 (SHA256SUMS della
# cartella riservata 02_Experiments/C1/20260928-C1-seed42-a80f6f9 e run report C1)
C1_MANIFEST_SHA256 = "86095da1531b1f24b1233c9668ac1ab37714978e444b313456732c028ccfb40c"

# file locale dei modelli legati alla macchina (repository di sviluppo; mai copiato in un pacchetto)
# (LIVENESSLAB_LOCAL_SCAN_PATTERNS lo sposta, per esempio nei test)
LOCAL_PATTERNS_FILE = Path(os.environ.get("LIVENESSLAB_LOCAL_SCAN_PATTERNS")
                           or Path(__file__).resolve().parents[1] / "config" / "local_scan_patterns.json")
LOCAL_LABEL = "operator or host name"
TOOL_LABEL = "tool reference"
LOCAL_KEYS = ("operator_or_host_name", "tool_reference")


def load_local_patterns(path=None, key: str = "operator_or_host_name") -> list:
    """Alternative regex di una chiave del file locale (lista di stringhe); [] se il file manca. Un file presente ma
    illeggibile o malformato ferma l'import: meglio un errore che una scansione più debole."""
    p = Path(path) if path is not None else LOCAL_PATTERNS_FILE
    if not p.is_file():
        return []
    doc = json.loads(p.read_text(encoding="utf-8"))
    alts = doc.get(key) if isinstance(doc, dict) else None
    if not isinstance(alts, list) or not alts or not all(isinstance(x, str) and x for x in alts):
        raise ValueError(f"{p.name}: '{key}' must be a non-empty list of regular expressions")
    return alts


def local_patterns_problem(path=None) -> str:
    """Per gli export con --strict: errore chiaro se il file locale manca (la scansione sarebbe solo generica)."""
    p = Path(path) if path is not None else LOCAL_PATTERNS_FILE
    if not p.is_file():
        return (f"the local scan patterns file (config/{p.name}) is missing: the scan would run with the generic rules only; "
                "restore it in the development repository before a strict export")
    return ""


def forbidden_text(local: list, tool: list = ()) -> list:
    """Regole del contenuto vietato: quelle generiche più, se ci sono, i modelli locali (secondo posto) e i nomi di
    strumenti (ultimo posto, senza distinzione di maiuscole): stesso ordine e stesse espressioni di export_m1.py quando
    il file locale è presente."""
    rules = list(GENERIC_TEXT)
    if local:
        rules.insert(1, (LOCAL_LABEL, re.compile("|".join(local))))
    if tool:
        rules.append((TOOL_LABEL, re.compile("|".join(tool), re.I)))
    return rules


GENERIC_TEXT = [
    ("absolute path", re.compile(r"/Us" r"ers/|/home/[a-z]|[A-Za-z]:\\+Users\\+|/priv" r"ate/(?:tmp|var)|/var/fol" r"ders/")),
    ("token or credential", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_\w{20,}|\bhf_[A-Za-z0-9]{20,}|\bsk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}"
                                       r"|xox[abprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|Bearer\s+[A-Za-z0-9._-]{20,}"
                                       r"|(?:password|passwd|secret|api_key|apikey|token)\s*[:=]\s*[\"'][^\"'\s]{8,}[\"']", re.I)),
]
LOCAL_PATTERNS = load_local_patterns()
TOOL_WORDS = load_local_patterns(key="tool_reference")
FORBIDDEN_TEXT = forbidden_text(LOCAL_PATTERNS, TOOL_WORDS)
# indirizzi IPv4 (ammessi solo quelli locali) e segni di lavoro non finito nei .md, come in export_m1.py
IPV4 = re.compile(r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\w.])")
IP_ALLOW = {"127.0.0.1", "0.0.0.0"}
MARKERS = re.compile(r"(?i:to" r"do)|T" r"BD|FIX" r"ME|<<(?!'EOF')")
