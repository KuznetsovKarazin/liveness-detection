"""
Esportazione dei risultati della valutazione in un file Excel (.xlsx) scritto a mano: un .xlsx è uno zip di file XML
(Office Open XML), quindi non serve alcuna libreria. Fogli: uno per dataset valutato (metriche per analizzatore),
"Confronto" (ACER ed EER per analizzatore × dataset), "ROC" (i punti delle curve, per tracciarle altrove) e "Info"
(dataset, licenze, note, convenzioni, data di generazione). Le stringhe sono "inline" (`t="inlineStr"`), i numeri
numeri: Excel, Numbers e LibreOffice li aprono senza avvisi.
"""
from __future__ import annotations

import io
import math
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
from xml.sax.saxutils import escape, quoteattr

from .core import registry
from . import evaluation

METRICS = [("apcer", "APCER"), ("bpcer", "BPCER"), ("acer", "ACER"), ("eer", "EER"), ("bpcer_at_apcer10", "BPCER@APCER10%"),
           ("auc", "AUC"), ("accuracy", "Accuracy")]
FAMILY = {"docente": "CNN del docente", "pretrained": "Pre-addestrato", "classico": "Classico spiegabile"}
REL = {"trained": "addestrato", "heuristic": "euristico", "untrained": "non addestrato", "zeroshot": "zero-shot", "descriptive": "descrittivo", "error": "errore"}


def _col(n: int) -> str:
    """Indice di colonna (0-based) → lettera Excel (A, B, …, AA)."""
    s = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _cell(ref: str, v: Any) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, bool):
        return f'<c r="{ref}" t="inlineStr"><is><t>{"sì" if v else "no"}</t></is></c>'
    if isinstance(v, float) and not math.isfinite(v):
        return ""                                  # NaN/inf non sono numeri validi in un foglio: cella vuota
    if isinstance(v, (int, float)):
        return f'<c r="{ref}"><v>{v}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{escape(str(v))}</t></is></c>'


def _sheet(rows: Sequence[Sequence[Any]], widths: Optional[Sequence[float]] = None) -> str:
    cols = ""
    if widths:
        cols = "<cols>" + "".join(f'<col min="{i + 1}" max="{i + 1}" width="{w}" customWidth="1"/>' for i, w in enumerate(widths)) + "</cols>"
    body = []
    for r, row in enumerate(rows, 1):
        cells = "".join(_cell(f"{_col(c)}{r}", v) for c, v in enumerate(row))
        body.append(f'<row r="{r}">{cells}</row>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'{cols}<sheetData>{"".join(body)}</sheetData></worksheet>')


def _safe_name(name: str, used: Optional[set] = None, fallback: str = "Foglio") -> str:
    """Nome di foglio valido per Excel: al massimo 31 caratteri, senza : \\ / ? * [ ], non vuoto, senza apostrofi agli
    estremi, unico tra quelli già usati (suffisso numerico)."""
    for ch in ':\\/?*[]':
        name = name.replace(ch, " ")
    name = name.strip().strip("'").strip()[:31] or fallback
    if used is not None:
        base, n = name, 2
        while name in used:
            suffix = f" ({n})"; name = base[:31 - len(suffix)] + suffix; n += 1
        used.add(name)
    return name


def build_workbook() -> bytes:
    """Il file .xlsx con tutte le valutazioni in cache."""
    datasets = {d["id"]: d for d in evaluation.describe_datasets()}
    summaries = evaluation.all_summaries()
    summaries = dict(sorted(summaries.items(), key=lambda kv: kv[0] == "samples"))   # le immagini di prova per ultime
    names = {a.id: a for a in registry.all()}
    sheets: List[tuple] = []
    used_names: set = {"Confronto", "ROC", "Info"}
    # un foglio per dataset
    head = ["Algoritmo", "Famiglia", "Affidabilità", "Addestrato su", "n"] + [m[1] for m in METRICS] + ["ms/img", "Errori", "Senza volto", "Nota", "Impronta"]
    for ds_id, sm in summaries.items():
        rows: List[List[Any]] = [head]
        ordered = sorted(sm["analyzers"].items(), key=lambda kv: ((1 if kv[1].get("note") else 0), getattr(names.get(kv[0]), "order", 99)))
        for aid, m in ordered:
            a = names.get(aid)
            base = [getattr(a, "name", aid), FAMILY.get(getattr(a, "family", ""), ""), REL.get(a.reliability(), "") if a else "", getattr(a, "subgroup", "") if a else "", m.get("n")]
            if m.get("note"):
                rows.append(base + [None] * len(METRICS) + [None, None, None, m["note"], None])
            else:
                rows.append(base + [m.get(k) for k, _ in METRICS] + [round(m["elapsed_ms_mean"], 1) if m.get("elapsed_ms_mean") is not None else None,
                                    m.get("n_errors"), m.get("n_noface"), None, m.get("fingerprint")])
        d = datasets.get(ds_id, {})
        rows += [[], ["Dataset", d.get("name", ds_id)], ["Fonte", d.get("source", "")], ["Licenza", d.get("license", "")], ["Nota", d.get("note", "")],
                 ["Aggiornato", sm.get("updated", "")], ["Convenzione", "punteggio = probabilità di attacco, soglia 0,5; APCER/BPCER/ACER/Accuracy alla soglia; EER, BPCER@APCER10% e AUC indipendenti dalla soglia"]]
        sheets.append((_safe_name(d.get("name") or ds_id, used_names, ds_id), rows, [34, 18, 14, 14, 6] + [10] * len(METRICS) + [8, 7, 10, 60, 40]))
    # confronto tra dataset
    ids = list(summaries)
    aids = sorted({aid for s in summaries.values() for aid, m in s["analyzers"].items() if m.get("acer") is not None}, key=lambda aid: getattr(names.get(aid), "order", 99))
    dn = lambda i: datasets.get(i, {}).get("name", i)  # noqa: E731
    rows = [["Algoritmo"] + [f"{dn(i)} · ACER" for i in ids] + [f"{dn(i)} · EER" for i in ids] + [f"{dn(i)} · n" for i in ids]]
    for aid in aids:
        ms = [summaries[i]["analyzers"].get(aid, {}) for i in ids]
        rows.append([getattr(names.get(aid), "name", aid)] + [m.get("acer") for m in ms] + [m.get("eer") for m in ms] + [m.get("n") for m in ms])
    rows += [[], ["Lettura", "Per riga: quanto un metodo regge quando cambia il dataset (cross-dataset). Per i modelli addestrati, la cella del dataset di addestramento è l'intra-dataset. Le colonne n dicono su quante immagini è calcolata ogni cella: confrontare solo celle con n uguale."]]
    sheets.append(("Confronto", rows, [34] + [18] * (3 * len(ids))))
    # punti delle curve ROC
    rows = [["Dataset", "Algoritmo", "BPCER (x)", "1 − APCER (y)"]]
    for i in ids:
        for aid, m in summaries[i]["analyzers"].items():
            for x, y in (m.get("roc") or []):
                rows.append([datasets.get(i, {}).get("name", i), getattr(names.get(aid), "name", aid), x, y])
    sheets.append(("ROC", rows, [24, 34, 12, 14]))
    # informazioni
    rows = [["LivenessLab · valutazione su dataset"], ["Generato il", datetime.now().strftime("%d/%m/%Y %H:%M")], [],
            ["Metriche", "ISO/IEC 30107-3: APCER (attacchi accettati), BPCER (bona fide rifiutati), ACER = media; EER; BPCER@APCER10%; AUC. Formule di evaluation_utils.py del docente con la BPCER@APCER corretta."],
            ["Soglia", "0,5 sul punteggio (probabilità di attacco); pareggio = bona fide"],
            ["Esclusioni", "Le CNN con lo split 80/20 del docente non sono valutate sul proprio dataset (le immagini di valutazione erano nel loro training)."], []]
    rows.append(["Dataset", "Reali", "Attacchi", "Fonte", "Licenza", "Nota"])
    for d in datasets.values():
        rows.append([d.get("name"), d.get("n_real"), d.get("n_attack"), d.get("source", ""), d.get("license", ""), d.get("note", "")])
    sheets.append(("Info", rows, [26, 8, 9, 60, 30, 80]))
    return _zip(sheets)


def _zip(sheets: List[tuple]) -> bytes:
    """Impacchetta i fogli in un .xlsx (zip con le parti minime di Office Open XML)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        n = len(sheets)
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="xml" ContentType="application/xml"/>'
                   '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                   + "".join(f'<Override PartName="/xl/worksheets/sheet{i + 1}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(n))
                   + '</Types>')
        z.writestr("_rels/.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                   '</Relationships>')
        z.writestr("xl/workbook.xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
                   + "".join(f'<sheet name={quoteattr(name)} sheetId="{i + 1}" r:id="rId{i + 1}"/>' for i, (name, _, _) in enumerate(sheets))
                   + '</sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   + "".join(f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i + 1}.xml"/>' for i in range(n))
                   + '</Relationships>')
        for i, (_, rows, widths) in enumerate(sheets):
            z.writestr(f"xl/worksheets/sheet{i + 1}.xml", _sheet(rows, widths))
    return buf.getvalue()
