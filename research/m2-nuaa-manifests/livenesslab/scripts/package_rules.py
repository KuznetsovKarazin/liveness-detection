"""
Costanti comuni ai pacchetti pubblici (M2 e successivi): regole del contenuto vietato e SHA-256 del manifest C1
consegnato (versione 1, congelata).

Copia dei valori di scripts/export_m1.py, che resta invariato: scripts/check_nuaa_manifests.py verifica che pattern,
flag ed etichette di FORBIDDEN_TEXT e C1_MANIFEST_SHA256 siano identici a quelli di export_m1.py. Le parole vietate
sono spezzate nel sorgente, così questo file supera lo stesso controllo quando entra in un pacchetto pubblico.
"""
import re

# SHA-256 del manifest C1 (results/c1/nuaa_manifest.csv) come registrato alla consegna di C1 (SHA256SUMS della
# cartella riservata 02_Experiments/C1/20260928-C1-seed42-a80f6f9 e run report C1)
C1_MANIFEST_SHA256 = "86095da1531b1f24b1233c9668ac1ab37714978e444b313456732c028ccfb40c"

AI_WORDS = ("cl" + "aude", "anth" + "ropic")
FORBIDDEN_TEXT = [
    ("absolute path", re.compile(r"/Us" r"ers/|/home/[a-z]|[A-Za-z]:\\+Users\\+|/priv" r"ate/(?:tmp|var)|/var/fol" r"ders/")),
    ("operator or server name", re.compile(r"vale" r"cass|srv-" r"tesi|Admini" r"strator|admin_" r"token|195\.32\.\d+\.\d+|100\.95\.\d+\.\d+")),
    ("token or credential", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_\w{20,}|\bhf_[A-Za-z0-9]{20,}|\bsk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}"
                                       r"|xox[abprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|Bearer\s+[A-Za-z0-9._-]{20,}"
                                       r"|(?:password|passwd|secret|api_key|apikey|token)\s*[:=]\s*[\"'][^\"'\s]{8,}[\"']", re.I)),
    ("AI tool reference", re.compile("|".join(AI_WORDS), re.I)),
]
# indirizzi IPv4 (ammessi solo quelli locali) e segni di lavoro non finito nei .md, come in export_m1.py
IPV4 = re.compile(r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\w.])")
IP_ALLOW = {"127.0.0.1", "0.0.0.0"}
MARKERS = re.compile(r"(?i:to" r"do)|T" r"BD|FIX" r"ME|<<(?!'EOF')")
