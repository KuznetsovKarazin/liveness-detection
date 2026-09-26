"""Rapporto degli utilizzi: chi ha aperto il sito, quando, da quale indirizzo e con quale browser.
Legge il log W3C di IIS del dominio e logs/usage.jsonl (scritto da usage.py) e produce un testo per giorno e per
indirizzo, escludendo il gestore (MY_IPS) e separando scansioni e bot. Usato dallo script
scripts/usage_report.py (dalla share) e dall'endpoint riservato /api/usage/report (sul server)."""
import json
import re
import socket
import subprocess
from collections import defaultdict
import os
from datetime import datetime, timedelta
from pathlib import Path

from .usage import USAGE_LOG

MY_IPS: set = set()                        # indirizzi del gestore noti (aggiungere qui quelli nuovi)
BOT_UA = re.compile(r"bot|crawl|spider|slurp|python|curl|wget|go-http|scan|zgrab|censys|masscan|nmap|headless|libwww|java/|okhttp|axios|node-fetch|http\.rb|dataprovider|expanse|paloalto|research", re.I)
APP_PATHS = ("/", "/ws", "/api/", "/static/", "/manifest.webmanifest", "/sw.js", "/favicon.ico")


def ua_short(ua: str) -> str:
    """Browser e sistema in due parole: 'Chrome 150 · macOS', 'Safari · iPhone'."""
    ua = ua.replace("+", " ")
    if not ua or ua == "-":
        return "?"
    os_ = ("iPhone" if "iPhone" in ua else "iPad" if "iPad" in ua else "Android" if "Android" in ua else
           "Windows" if "Windows" in ua else "macOS" if "Macintosh" in ua else "Linux" if "Linux" in ua else "?")
    m = (re.search(r"(Edg|EdgiOS)/(\d+)", ua) or re.search(r"(OPR)/(\d+)", ua) or re.search(r"(SamsungBrowser)/(\d+)", ua)
         or re.search(r"(Firefox|FxiOS)/(\d+)", ua) or re.search(r"(CriOS)/(\d+)", ua) or re.search(r"(Chrome)/(\d+)", ua)
         or re.search(r"Version/(\d+)[^ ]* (Safari)", ua))
    if m:
        name, ver = (m.group(2), m.group(1)) if m.group(1).isdigit() else (m.group(1), m.group(2))
        name = {"Edg": "Edge", "EdgiOS": "Edge", "OPR": "Opera", "CriOS": "Chrome", "FxiOS": "Firefox"}.get(name, name)
        return f"{name} {ver} · {os_}"
    return f"{ua[:30]} · {os_}"


def rdns(ip: str, cache: dict) -> str:
    if ip not in cache:
        try:
            socket.setdefaulttimeout(2.0)
            cache[ip] = socket.gethostbyaddr(ip)[0]
        except Exception:  # noqa: BLE001
            cache[ip] = ""
    return cache[ip]


def whois(ip: str, cache: dict) -> str:
    """Rete e paese dal registro (RIPE, ARIN…): 'TELECOM-ITALIA IT'. Vuoto se il comando manca o non risponde."""
    if ip not in cache:
        cache[ip] = ""
        try:
            out = subprocess.run(["whois", ip], capture_output=True, text=True, timeout=8).stdout
            net = next((l.split(":", 1)[1].strip() for l in out.splitlines() if re.match(r"(?i)^(netname|org-name|orgname|descr):", l)), "")
            country = next((l.split(":", 1)[1].strip() for l in out.splitlines() if re.match(r"(?i)^country:", l)), "")
            cache[ip] = " ".join(x for x in (net[:32], country) if x)
        except (OSError, subprocess.TimeoutExpired):
            pass
    return cache[ip]


def read_iis(folder: Path, since: datetime | None):
    rows = []
    for f in sorted(folder.glob("*.log")):
        fields = None
        with f.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("#Fields:"):
                    fields = line.split()[1:]; continue
                if line.startswith("#") or not fields:
                    continue
                parts = line.rstrip("\n").split(" ")
                if len(parts) < len(fields):
                    continue
                r = dict(zip(fields, parts))
                try:
                    ts = datetime.strptime(r["date"] + " " + r["time"], "%Y-%m-%d %H:%M:%S")
                except (KeyError, ValueError):
                    continue
                if since and ts < since:
                    continue
                rows.append({"ts": ts, "ip": r.get("c-ip", "?"), "ua": r.get("cs(User-Agent)", "-"), "path": r.get("cs-uri-stem", ""),
                             "status": r.get("sc-status", ""), "bytes": int(r.get("sc-bytes", "0") or 0), "ms": int(r.get("time-taken", "0") or 0)})
    return rows


def read_usage(path: Path, since: datetime | None):
    ev = []
    if not path.exists():
        return ev
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
                ts = datetime.strptime(d["ts"], "%Y-%m-%dT%H:%M:%S")
            except (ValueError, KeyError):
                continue
            if since and ts < since:
                continue
            d["ts"] = ts; ev.append(d)
    return ev



def build_report(rows, usage, mine: set, days: int, show_all: bool = False, use_rdns: bool = True, use_whois: bool = False) -> str:
    """Testo del rapporto a partire dalle righe IIS e dagli eventi dell'app."""
    out = []
    cache: dict = {}; wcache: dict = {}

    def is_me(ip, ua):
        return ip in mine
    per_ip = defaultdict(lambda: {"hits": 0, "pages": 0, "ws": 0, "ws_bytes": 0, "first": None, "last": None, "uas": set(), "app": 0, "days": set()})
    for r in rows:
        p = per_ip[r["ip"]]
        p["hits"] += 1; p["uas"].add(r["ua"]); p["days"].add(r["ts"].date())
        p["first"] = min(p["first"] or r["ts"], r["ts"]); p["last"] = max(p["last"] or r["ts"], r["ts"])
        if r["path"].startswith(APP_PATHS) or r["path"] == "/":
            p["app"] += 1
        if r["path"] == "/" and r["status"] == "200":
            p["pages"] += 1
        if r["path"] == "/ws" and r["status"] == "101":
            p["ws"] += 1; p["ws_bytes"] += r["bytes"]
    acts = defaultdict(lambda: defaultdict(int))
    for e in usage:
        acts[e["ip"]][e["kind"]] += 1

    def kind_of(ip, p):
        if any(is_me(ip, u) for u in p["uas"]) and all(is_me(ip, u) or BOT_UA.search(u) for u in p["uas"]):
            return "me"
        if p["app"] == 0 or all(BOT_UA.search(u) for u in p["uas"]) or (p["pages"] == 0 and p["ws"] == 0 and p["hits"] < 3):
            return "bot"
        return "visitor"
    groups = defaultdict(list)
    for ip, p in per_ip.items():
        groups[kind_of(ip, p)].append((ip, p))
    periodo = f"ultimi {days} giorni" if days > 0 else "tutto lo storico"
    out.append(f"LivenessLab · utilizzi ({periodo}) · {len(rows)} richieste HTTP, {len(per_ip)} indirizzi, {len(usage)} eventi dell'app")
    out.append(f"  gestore ({', '.join(sorted(mine))}): {sum(p['hits'] for _, p in groups['me'])} richieste, {sum(p['pages'] for _, p in groups['me'])} aperture della pagina, {sum(p['ws'] for _, p in groups['me'])} sessioni")
    out.append(f"  scansioni e bot: {len(groups['bot'])} indirizzi, {sum(p['hits'] for _, p in groups['bot'])} richieste" + ("" if show_all else " (nascosti)"))
    out.append("")
    out.append("VISITATORI (esclusi gestore, bot e scansioni)")
    if not groups["visitor"]:
        out.append("  nessuno")
    else:
        out.append(f"  {'giorno':10s} {'ora':11s} {'indirizzo':16s} {'pagine':>6s} {'sess.':>5s} {'analisi':>7s} {'valut.':>6s}  browser · sistema / nome")
    for ip, p in sorted(groups["visitor"], key=lambda x: x[1]["last"], reverse=True):
        name = rdns(ip, cache) if use_rdns else ""
        if use_whois:
            name = " · ".join(x for x in (name, whois(ip, wcache)) if x)
        ua = " | ".join(sorted({ua_short(u) for u in p["uas"]}))
        ora = f"{p['first']:%H:%M}–{p['last']:%H:%M}" if p["first"].date() == p["last"].date() else f"{p['first']:%d/%m %H:%M}→{p['last']:%d/%m %H:%M}"
        # sessioni con lavoro: WebSocket che hanno trasmesso più di 100 KB (eventi di un'analisi o valutazione)
        out.append(f"  {p['last']:%Y-%m-%d} {ora:11s} {ip:16s} {p['pages']:>6d} {p['ws']:>5d} {acts[ip]['analyze']:>7d} {acts[ip]['evaluate']:>6d}  {ua}{(' / ' + name) if name else ''}"
                   + (f"   [{len(p['days'])} giorni]" if len(p["days"]) > 1 else "") + (f"   ~{p['ws_bytes'] // 1024} KB di eventi" if p["ws_bytes"] > 100_000 else ""))
    if show_all and groups["bot"]:
        out.append("")
        out.append("SCANSIONI E BOT")
        for ip, p in sorted(groups["bot"], key=lambda x: x[1]["hits"], reverse=True)[:40]:
            out.append(f"  {p['last']:%Y-%m-%d %H:%M} {ip:16s} {p['hits']:>5d} richieste  {ua_short(next(iter(p['uas'])))[:40]}")
    if not usage:
        out.append("")
        out.append("(Analisi e valutazioni per indirizzo compaiono quando il server scrive logs/usage.jsonl: versione dal 26/09.)")
    return "\n".join(out)


def report(days: int = 1, show_all: bool = False, mine: set | None = None, iis_dir: Path | None = None, usage_path: Path | None = None,
           use_rdns: bool = True, use_whois: bool = False) -> str:
    """Rapporto completo dalle sorgenti indicate (default: quelle del server)."""
    iis_dir = iis_dir or Path(os.environ.get("LIVENESSLAB_IIS_LOG_DIR", r"C:\inetpub\vhosts\tesi.valeriocassano.com\logs\iis\W3SVC47"))
    usage_path = usage_path or USAGE_LOG
    if not iis_dir.exists():
        raise FileNotFoundError(f"log IIS non trovato in {iis_dir}")
    since = datetime.now() - timedelta(days=days) if days > 0 else None
    return build_report(read_iis(iis_dir, since), read_usage(usage_path, since), mine or MY_IPS, days, show_all, use_rdns, use_whois)
