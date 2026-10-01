"""
Controlli delle metriche PAD con valori attesi indipendenti (M1, WP1).

Ogni valore atteso è scritto a mano come frazione (`fractions.Fraction`) e derivato in metric_cases.md,
oppure, per il caso reale, calcolato da un'implementazione di riferimento scritta qui (conteggi diretti, soglie
ordinate, Mann–Whitney per la AUC): mai scikit-learn, mai il modulo in prova. Le convenzioni sono quelle di
metrics.md, sezione "Conventions". Cache e tabella C1 del caso reale si leggono da LIVENESSLAB_RESULTS_DIR
(eval/nuaa.json, c1/nuaa_table.md); se mancano, quei casi vengono saltati e il riepilogo lo dice
("RESULT: PASS (N cases skipped: restricted inputs not found)"); con --require-restricted ogni salto è un fallimento.
Con la tabella presente, le sue righe numeriche devono essere esattamente gli analizzatori della cache con punteggi
finiti (30, senza duplicati né righe sconosciute).

  .venv/bin/python scripts/check_metrics.py                          # rapporto su stdout, uscita 1 se un caso fallisce
  .venv/bin/python scripts/check_metrics.py --report path/report.txt # anche su file
  .venv/bin/python scripts/check_metrics.py --require-restricted     # uscita 1 anche se mancano cache o tabella C1

Solo libreria standard (unittest); il modulo in prova è tesi_app.evaluation (non importa TensorFlow).
"""
from __future__ import annotations

import argparse
import bisect
import csv
import io
import json
import math
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from tesi_app import evaluation  # noqa: E402
from tesi_app.paths import RESULTS  # noqa: E402

# input riservati del caso reale: cartella dei risultati (LIVENESSLAB_RESULTS_DIR), come eval_dataset.py
NUAA_CACHE = RESULTS / "eval" / "nuaa.json"
NUAA_TABLE = RESULTS / "c1" / "nuaa_table.md"
EXPECTED_REAL_ROWS = 30                             # righe numeriche della tabella C1 di NUAA (analizzatori con punteggi)
# con --require-restricted un input riservato mancante è un fallimento, non un salto
REQUIRE_RESTRICTED = False

# Tolleranza assoluta per i valori float del modulo rispetto alla frazione esatta. Una divisione k/n in doppia precisione
# è arrotondata correttamente (errore relativo <= 2^-53, circa 1e-16) e medie e somme aggiungono pochi ulp: 1e-12 è molto
# sopra questo rumore e molto sotto la più piccola differenza significativa possibile qui (1/(n_attack*n_bona_fide),
# almeno 1/22500 ≈ 4e-5 sui 300 di NUAA), quindi non può nascondere un errore vero. Le soglie EER si confrontano invece
# per uguaglianza esatta (sono punteggi presi dall'input) e i conteggi come interi.
ABS_TOL = 1e-12

METRICS = ("n", "n_bona_fide", "n_attack", "apcer", "bpcer", "acer", "accuracy",
           "eer", "eer_threshold", "apcer_at_eer", "bpcer_at_eer", "auc", "bpcer_at_apcer10")
F = Fraction

# righe del rapporto: (caso, metrica, atteso, ottenuto, esito)
REPORT_ROWS: list = []
CASE_INFO: dict = {}


# ----------------------------------------------------------------------------- implementazione di riferimento

def ref_counts(y, s, threshold):
    """Conteggi diretti con la regola operativa `s > soglia` (pareggio = bona fide)."""
    bona = [x for yi, x in zip(y, s) if yi == 0]
    att = [x for yi, x in zip(y, s) if yi == 1]
    rejected_bona = sum(1 for x in bona if x > threshold)
    accepted_att = sum(1 for x in att if not x > threshold)
    return bona, att, rejected_bona, accepted_att


def ref_roc(bona, att):
    """Curva ROC completa come roc_curve(drop_intermediate=False): soglia iniziale +inf, poi ogni punteggio distinto
    in ordine decrescente; alla soglia t è attacco chi ha s >= t. Punti: (soglia, fp, fn) come conteggi."""
    thresholds = [math.inf] + sorted(set(bona) | set(att), reverse=True)
    pts = []
    for t in thresholds:
        fp = sum(1 for x in bona if x >= t)        # bona fide rifiutati
        fn = sum(1 for x in att if not x >= t)     # attacchi accettati
        pts.append((t, fp, fn))
    return pts


def ref_eer(bona, att):
    """EER senza interpolazione: punto con |BPCER - APCER| minimo, primo in ordine di soglia decrescente.
    Il confronto è su interi (|fp*n_a - fn*n_b|), così i pareggi sono esatti."""
    nb, na = len(bona), len(att)
    best = None
    for t, fp, fn in ref_roc(bona, att):
        key = abs(fp * na - fn * nb)
        if best is None or key < best[0]:
            best = (key, t, fp, fn)
    _, t, fp, fn = best
    return (F(fp, nb) + F(fn, na)) / 2, (None if math.isinf(t) else t)


def ref_eer_interpolated(bona, att):
    """EER diagnostico: interpolazione lineare tra i due punti adiacenti della ROC in cui BPCER - APCER cambia segno."""
    nb, na = len(bona), len(att)
    pts = [(F(fp, nb), F(fn, na)) for _, fp, fn in ref_roc(bona, att)]
    for (b0, a0), (b1, a1) in zip(pts, pts[1:]):
        d0, d1 = b0 - a0, b1 - a1
        if d0 == 0:
            return b0
        if d0 < 0 < d1:
            lam = -d0 / (d1 - d0)
            return b0 + lam * (b1 - b0)
        if d1 == 0:
            return b1
    return None


def ref_auc(bona, att):
    """AUC come statistica di Mann–Whitney: coppie (attacco, bona fide) con attacco > bona fide, più metà dei pareggi."""
    sb = sorted(bona)
    twice_u = 0
    for x in att:
        lo, hi = bisect.bisect_left(sb, x), bisect.bisect_right(sb, x)
        twice_u += 2 * lo + (hi - lo)
    return F(twice_u, 2 * len(att) * len(bona))


def ref_bpcer_at_apcer10(bona, att):
    """Soglia = punteggio di attacco all'indice floor(n_attack/10) dell'elenco crescente; BPCER = quota di bona fide >= soglia."""
    if len(att) < 10:                               # n_attack * 0,10 < 1
        return None
    t = sorted(att)[len(att) // 10]
    return F(sum(1 for x in bona if x >= t), len(bona))


def ref_metrics(y, s, threshold=0.5):
    """Tutte le metriche secondo le convenzioni di metrics.md, in frazioni esatte."""
    bona, att, rej, acc_att = ref_counts(y, s, threshold)
    nb, na, n = len(bona), len(att), len(y)
    apcer = F(acc_att, na) if na else None
    bpcer = F(rej, nb) if nb else None
    out = {"n": n, "n_bona_fide": nb, "n_attack": na, "threshold": threshold, "apcer": apcer, "bpcer": bpcer,
           "acer": (apcer + bpcer) / 2 if apcer is not None and bpcer is not None else None,
           "accuracy": F((nb - rej) + (na - acc_att), n) if n else None,
           "eer": None, "eer_threshold": None, "apcer_at_eer": None, "bpcer_at_eer": None,
           "auc": None, "bpcer_at_apcer10": None}
    if nb and na:
        constant = max(s) - min(s) == 0             # costanza esatta, nessuna tolleranza
        out["eer"], out["eer_threshold"] = ref_eer(bona, att)
        if out["eer_threshold"] is not None:
            # errori al punto ROC dell'EER, regola inclusiva s >= soglia EER (come la curva)
            t = out["eer_threshold"]
            out["apcer_at_eer"] = F(sum(1 for x in att if not x >= t), na)
            out["bpcer_at_eer"] = F(sum(1 for x in bona if x >= t), nb)
        out["auc"] = ref_auc(bona, att)
        out["bpcer_at_apcer10"] = None if constant else ref_bpcer_at_apcer10(bona, att)
    return out


# ----------------------------------------------------------------------------- casi sintetici (valori scritti a mano)

CASES = {
    "perfect": dict(
        desc="perfect separation",
        y=[0, 0, 0, 1, 1, 1], s=[0.1, 0.2, 0.3, 0.7, 0.8, 0.9], threshold=0.5,
        expected=dict(n=6, n_bona_fide=3, n_attack=3, apcer=F(0, 3), bpcer=F(0, 3), acer=F(0), accuracy=F(6, 6),
                      eer=F(0), eer_threshold=0.7, auc=F(9, 9), bpcer_at_apcer10=None),
        eer_interpolated=F(0)),
    "inverted": dict(
        desc="inverted scores (every decision wrong)",
        y=[0, 0, 0, 1, 1, 1], s=[0.9, 0.8, 0.7, 0.3, 0.2, 0.1], threshold=0.5,
        expected=dict(n=6, n_bona_fide=3, n_attack=3, apcer=F(3, 3), bpcer=F(3, 3), acer=F(1), accuracy=F(0, 6),
                      eer=F(1), eer_threshold=0.7, auc=F(0, 9), bpcer_at_apcer10=None),
        eer_interpolated=F(1)),
    "constant_two_classes": dict(
        desc="constant scores 0.7 with both classes",
        y=[0, 0, 1, 1], s=[0.7, 0.7, 0.7, 0.7], threshold=0.5,
        expected=dict(n=4, n_bona_fide=2, n_attack=2, apcer=F(0, 2), bpcer=F(2, 2), acer=F(1, 2), accuracy=F(2, 4),
                      eer=F(1, 2), eer_threshold=None, auc=F(1, 2), bpcer_at_apcer10=None),
        eer_interpolated=F(1, 2)),
    "constant_ten_attacks": dict(
        desc="constant scores 0.7 with 2 bona fide and 10 attacks (BPCER@APCER10 undefined by the constant rule)",
        y=[0] * 2 + [1] * 10, s=[0.7] * 12, threshold=0.5,
        expected=dict(n=12, n_bona_fide=2, n_attack=10, apcer=F(0, 10), bpcer=F(2, 2), acer=F(1, 2),
                      accuracy=F(10, 12), eer=F(1, 2), eer_threshold=None, auc=F(10, 20), bpcer_at_apcer10=None),
        eer_interpolated=F(1, 2)),
    "constant_only_bona_fide": dict(
        desc="constant scores 0.7, bona fide only",
        y=[0, 0, 0], s=[0.7, 0.7, 0.7], threshold=0.5,
        expected=dict(n=3, n_bona_fide=3, n_attack=0, apcer=None, bpcer=F(3, 3), acer=None, accuracy=F(0, 3),
                      eer=None, eer_threshold=None, auc=None, bpcer_at_apcer10=None)),
    "constant_only_attacks": dict(
        desc="constant scores 0.3, attacks only",
        y=[1, 1], s=[0.3, 0.3], threshold=0.5,
        expected=dict(n=2, n_bona_fide=0, n_attack=2, apcer=F(2, 2), bpcer=None, acer=None, accuracy=F(0, 2),
                      eer=None, eer_threshold=None, auc=None, bpcer_at_apcer10=None)),
    "constant_at_threshold": dict(
        desc="constant scores equal to the threshold 0.5, both classes (tie = bona fide)",
        y=[0, 0, 1, 1], s=[0.5, 0.5, 0.5, 0.5], threshold=0.5,
        expected=dict(n=4, n_bona_fide=2, n_attack=2, apcer=F(2, 2), bpcer=F(0, 2), acer=F(1, 2), accuracy=F(2, 4),
                      eer=F(1, 2), eer_threshold=None, auc=F(1, 2), bpcer_at_apcer10=None),
        eer_interpolated=F(1, 2)),
    "near_constant": dict(
        desc="near-constant scores [0.5, 0.5000001], labels [0, 1]",
        y=[0, 1], s=[0.5, 0.5000001], threshold=0.5,
        expected=dict(n=2, n_bona_fide=1, n_attack=1, apcer=F(0, 1), bpcer=F(0, 1), acer=F(0), accuracy=F(2, 2),
                      eer=F(0), eer_threshold=0.5000001, auc=F(1, 1), bpcer_at_apcer10=None),
        eer_interpolated=F(0)),
    "ties_at_threshold": dict(
        desc="scores exactly equal to the threshold 0.5 in both classes (rule s > t)",
        y=[0, 0, 0, 0, 1, 1, 1, 1], s=[0.2, 0.5, 0.5, 0.8, 0.3, 0.5, 0.9, 0.95], threshold=0.5,
        expected=dict(n=8, n_bona_fide=4, n_attack=4, apcer=F(2, 4), bpcer=F(1, 4), acer=F(3, 8), accuracy=F(5, 8),
                      eer=F(3, 8), eer_threshold=0.8, auc=F(11, 16), bpcer_at_apcer10=None),
        eer_interpolated=F(5, 12)),
    "only_bona_fide": dict(
        desc="no attack samples",
        y=[0, 0, 0], s=[0.2, 0.6, 0.4], threshold=0.5,
        expected=dict(n=3, n_bona_fide=3, n_attack=0, apcer=None, bpcer=F(1, 3), acer=None, accuracy=F(2, 3),
                      eer=None, eer_threshold=None, auc=None, bpcer_at_apcer10=None)),
    "only_attacks": dict(
        desc="no bona fide samples",
        y=[1, 1, 1, 1], s=[0.2, 0.6, 0.7, 0.9], threshold=0.5,
        expected=dict(n=4, n_bona_fide=0, n_attack=4, apcer=F(1, 4), bpcer=None, acer=None, accuracy=F(3, 4),
                      eer=None, eer_threshold=None, auc=None, bpcer_at_apcer10=None)),
    "empty": dict(
        desc="empty input",
        y=[], s=[], threshold=0.5,
        expected=dict(n=0, n_bona_fide=0, n_attack=0, apcer=None, bpcer=None, acer=None, accuracy=None,
                      eer=None, eer_threshold=None, auc=None, bpcer_at_apcer10=None)),
    "bpcer10_five_attacks": dict(
        desc="BPCER@APCER10 with 5 attacks (undefined)",
        y=[0] * 4 + [1] * 5, s=[0.05, 0.15, 0.25, 0.35, 0.1, 0.2, 0.3, 0.4, 0.5], threshold=0.5,
        expected=dict(n_bona_fide=4, n_attack=5, bpcer_at_apcer10=None)),
    "bpcer10_ten_attacks": dict(
        desc="BPCER@APCER10 with 10 attacks (worked example of metrics.md)",
        y=[0] * 4 + [1] * 10,
        s=[0.05, 0.15, 0.25, 0.35, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], threshold=0.5,
        expected=dict(n_bona_fide=4, n_attack=10, bpcer_at_apcer10=F(2, 4))),
    "bpcer10_twenty_attacks_tie": dict(
        desc="BPCER@APCER10 with 20 attacks, bona fide score equal to the quantile threshold",
        y=[0] * 4 + [1] * 20,
        s=[0.1, 0.15, 0.3, 0.9,
           0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
           0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0], threshold=0.5,
        expected=dict(n_bona_fide=4, n_attack=20, bpcer_at_apcer10=F(3, 4))),
    "bpcer10_nine_attacks": dict(
        desc="BPCER@APCER10 with 9 attacks (boundary: 9 * 0.10 < 1, undefined)",
        y=[0] * 4 + [1] * 9, s=[0.05, 0.15, 0.25, 0.35, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], threshold=0.5,
        expected=dict(n_bona_fide=4, n_attack=9, bpcer_at_apcer10=None)),
    "bpcer10_fifteen_attacks_floor": dict(
        desc="BPCER@APCER10 with 15 attacks: floor(1.5) = 1, while ceil/round would give index 2 and another value",
        y=[0] * 4 + [1] * 15,
        s=[0.02, 0.12, 0.5, 0.9,
           0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75], threshold=0.5,
        expected=dict(n_bona_fide=4, n_attack=15, bpcer_at_apcer10=F(3, 4))),
    "stepped_roc": dict(
        desc="tied scores inside both classes (stepped ROC)",
        y=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1], s=[0.1, 0.4, 0.4, 0.6, 0.6, 0.4, 0.6, 0.8, 0.8, 0.8], threshold=0.5,
        expected=dict(n=10, n_bona_fide=5, n_attack=5, apcer=F(1, 5), bpcer=F(2, 5), acer=F(3, 10), accuracy=F(7, 10),
                      eer=F(3, 10), eer_threshold=0.6, apcer_at_eer=F(1, 5), bpcer_at_eer=F(2, 5),
                      auc=F(21, 25), bpcer_at_apcer10=None),
        eer_interpolated=F(4, 15)),
    "eer_tie_break": dict(
        desc="two ROC points with the same |BPCER - APCER| (first in decreasing threshold order wins)",
        y=[0, 0, 1, 1, 1], s=[0.4, 0.7, 0.3, 0.6, 0.9], threshold=0.5,
        expected=dict(n=5, n_bona_fide=2, n_attack=3, apcer=F(1, 3), bpcer=F(1, 2), acer=F(5, 12), accuracy=F(3, 5),
                      eer=F(7, 12), eer_threshold=0.7, auc=F(3, 6), bpcer_at_apcer10=None),
        eer_interpolated=F(1, 2)),
    "eer_tie_float_key": dict(
        desc="EER tie |1/3 - 1/2| = |2/3 - 1/2| = 1/6 at thresholds 0.2 and 0.15: exact integer key picks 0.2 (5/12); "
             "a float key |fpr - fnr| would pick 0.15 (7/12)",
        y=[0, 0, 0, 1, 1], s=[0.15, 0.0, 0.45, 0.2, 0.1], threshold=0.5,
        expected=dict(n=5, n_bona_fide=3, n_attack=2, apcer=F(2, 2), bpcer=F(0, 3), acer=F(1, 2), accuracy=F(3, 5),
                      eer=F(5, 12), eer_threshold=0.2, apcer_at_eer=F(1, 2), bpcer_at_eer=F(1, 3),
                      auc=F(3, 6), bpcer_at_apcer10=None),
        eer_interpolated=F(1, 2)),
    "default_threshold": dict(
        desc="compute_metrics called WITHOUT the threshold argument: the default must be 0.5 "
             "(a bona fide at 0.495 and an attack tied at 0.5 change side with another default)",
        y=[0, 0, 1, 1], s=[0.2, 0.495, 0.5, 0.8], threshold=None,
        expected=dict(n=4, n_bona_fide=2, n_attack=2, threshold=0.5, apcer=F(1, 2), bpcer=F(0, 2), acer=F(1, 4),
                      accuracy=F(3, 4))),
}


# ----------------------------------------------------------------------------- confronto e registrazione

def fmt(v):
    if v is None:
        return "None"
    if isinstance(v, Fraction):
        return f"{v.numerator}/{v.denominator} ({float(v)!r})" if v.denominator != 1 else f"{v.numerator} ({float(v)!r})"
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return repr(v)


def matches(expected, obtained):
    """None atteso = None esatto; frazione = float entro ABS_TOL; float (soglia) = uguaglianza esatta; intero = uguale."""
    if expected is None:
        return obtained is None
    if obtained is None or isinstance(obtained, bool):
        return False
    if isinstance(expected, Fraction):
        try:
            return abs(float(obtained) - float(expected)) <= ABS_TOL
        except (TypeError, ValueError):
            return False
    if isinstance(expected, float):
        return isinstance(obtained, (float, np.floating)) and float(obtained) == expected
    if isinstance(expected, int):
        return isinstance(obtained, (int, np.integer)) and int(obtained) == expected
    return expected == obtained


def obtained_value(m, key):
    # il modulo chiama n_real il numero di bona fide; si accetta anche n_bona_fide (nome delle convenzioni)
    if key == "n_bona_fide":
        return m.get("n_bona_fide", m.get("n_real"))
    if key in ("apcer_at_eer", "bpcer_at_eer"):
        return m.get(key)                           # chiavi presenti solo quando la soglia EER è definita
    return m.get(key, "<missing key>")


class MetricCase(unittest.TestCase):
    """Base: registra ogni confronto nel rapporto e fallisce alla fine del caso se almeno uno non torna."""

    def setUp(self):
        self._failed = []

    def check(self, case, metric, expected, obtained):
        ok = matches(expected, obtained)
        REPORT_ROWS.append((case, metric, fmt(expected), fmt(obtained), "PASS" if ok else "FAIL"))
        if not ok:
            self._failed.append(f"{metric}: expected {fmt(expected)}, obtained {fmt(obtained)}")

    def note_error(self, case, metric, message):
        REPORT_ROWS.append((case, metric, "-", message, "FAIL"))
        self._failed.append(f"{metric}: {message}")

    def finish(self, case):
        if self._failed:
            self.fail(f"{case}: " + "; ".join(self._failed))

    def run_synthetic(self, name):
        c = CASES[name]
        CASE_INFO[name] = f"{c['desc']} | y={c['y']} s={c['s']} threshold={c['threshold']}"
        y, s = np.array(c["y"], dtype=int), np.array(c["s"], dtype=float)
        # threshold None = chiamata senza l'argomento: si verifica il valore di default del modulo
        m = evaluation.compute_metrics(y, s) if c["threshold"] is None else evaluation.compute_metrics(y, s, c["threshold"])
        for key, exp in c["expected"].items():
            self.check(name, key, exp, obtained_value(m, key))
        self.finish(name)


# ----------------------------------------------------------------------------- test

class TestSynthetic(MetricCase):
    def test_01_perfect(self):
        self.run_synthetic("perfect")

    def test_02_inverted(self):
        self.run_synthetic("inverted")

    def test_03_constant_two_classes(self):
        self.run_synthetic("constant_two_classes")

    def test_03b_constant_ten_attacks(self):
        self.run_synthetic("constant_ten_attacks")

    def test_03c_constant_only_bona_fide(self):
        self.run_synthetic("constant_only_bona_fide")

    def test_03d_constant_only_attacks(self):
        self.run_synthetic("constant_only_attacks")

    def test_03e_constant_at_threshold(self):
        self.run_synthetic("constant_at_threshold")

    def test_04_near_constant(self):
        self.run_synthetic("near_constant")

    def test_05_ties_at_threshold(self):
        self.run_synthetic("ties_at_threshold")

    def test_06_only_bona_fide(self):
        self.run_synthetic("only_bona_fide")

    def test_07_only_attacks(self):
        self.run_synthetic("only_attacks")

    def test_08_empty(self):
        self.run_synthetic("empty")

    def test_09_bpcer10_five_attacks(self):
        self.run_synthetic("bpcer10_five_attacks")

    def test_10_bpcer10_ten_attacks(self):
        self.run_synthetic("bpcer10_ten_attacks")

    def test_10b_bpcer10_nine_attacks(self):
        self.run_synthetic("bpcer10_nine_attacks")

    def test_10c_bpcer10_fifteen_attacks_floor(self):
        self.run_synthetic("bpcer10_fifteen_attacks_floor")

    def test_11_bpcer10_twenty_attacks_tie(self):
        self.run_synthetic("bpcer10_twenty_attacks_tie")

    def test_12_stepped_roc(self):
        self.run_synthetic("stepped_roc")

    def test_13_eer_tie_break(self):
        self.run_synthetic("eer_tie_break")

    def test_13c_eer_tie_float_key(self):
        self.run_synthetic("eer_tie_float_key")

    def test_13d_default_threshold(self):
        self.run_synthetic("default_threshold")

    def test_13b_compact_roc(self):
        # out["roc"]: punti (BPCER, 1 - APCER) della ROC, arrotondati a 4 decimali, più [1.0, 1.0] in coda.
        # Caso scelto con 6 punti (< 60, passo 1) e senza tre punti consecutivi allineati, così lo scarto dei punti
        # intermedi di roc_curve (drop_intermediate=True) non toglie nulla e la lista è deterministica.
        name = "compact_roc"
        c = CASES["eer_tie_break"]
        CASE_INFO[name] = f"out['roc'] on the eer_tie_break data | y={c['y']} s={c['s']}"
        m = evaluation.compute_metrics(np.array(c["y"], dtype=int), np.array(c["s"], dtype=float), c["threshold"])
        # soglie +inf, 0.9, 0.7, 0.6, 0.4, 0.3 -> (fp/2, tp/3): 1/3 -> 0.3333, 2/3 -> 0.6667
        expected = [[0.0, 0.0], [0.0, 0.3333], [0.5, 0.3333], [0.5, 0.6667], [1.0, 0.6667], [1.0, 1.0], [1.0, 1.0]]
        got = m.get("roc")
        got_list = [[float(a), float(b)] for a, b in got] if isinstance(got, (list, tuple)) else got
        ok = got_list == expected
        REPORT_ROWS.append((name, "roc", str(expected), str(got_list), "PASS" if ok else "FAIL"))
        if not ok:
            self._failed.append(f"roc: expected {expected}, obtained {got_list}")
        self.finish(name)


class TestInclusiveDecision(MetricCase):
    """Regola diagnostica `s >= t`. Interfaccia attesa: staticmethod di
    BiometricMetrics, `decisions_at(y_scores, threshold, inclusive=False)` -> array di 0/1 (1 = attacco), con
    `s > threshold` se inclusive=False (regola operativa) e `s >= threshold` se inclusive=True (solo diagnostica)."""

    def test_14_decisions_at_inclusive(self):
        name = "decisions_at"
        c = CASES["ties_at_threshold"]
        CASE_INFO[name] = f"ties_at_threshold data, rule s >= t vs s > t | y={c['y']} s={c['s']} threshold=0.5"
        fn = getattr(evaluation.BiometricMetrics, "decisions_at", None)
        if fn is None:
            self.note_error(name, "decisions_at", "BiometricMetrics.decisions_at is not available")
            return self.finish(name)
        y = np.array(c["y"], dtype=int)
        s = np.array(c["s"], dtype=float)
        expected = {
            # s >= 0,5: bona fide rifiutati {0.5, 0.5, 0.8} = 3/4, attacchi accettati {0.3} = 1/4
            True: ([0, 1, 1, 1, 0, 1, 1, 1], F(1, 4), F(3, 4)),
            # s > 0,5: bona fide rifiutati {0.8} = 1/4, attacchi accettati {0.3, 0.5} = 2/4
            False: ([0, 0, 0, 1, 0, 0, 1, 1], F(2, 4), F(1, 4)),
        }
        for inclusive, (dec_exp, apcer_exp, bpcer_exp) in expected.items():
            tag = "inclusive" if inclusive else "strict"
            try:
                dec = [int(v) for v in np.asarray(fn(s, 0.5, inclusive=inclusive)).ravel()]
            except Exception as exc:  # noqa: BLE001
                self.note_error(name, f"decisions[{tag}]", f"{type(exc).__name__}: {exc}")
                continue
            REPORT_ROWS.append((name, f"decisions[{tag}]", str(dec_exp), str(dec), "PASS" if dec == dec_exp else "FAIL"))
            if dec != dec_exp:
                self._failed.append(f"decisions[{tag}]: expected {dec_exp}, obtained {dec}")
            # APCER e BPCER ricavati dalle decisioni con conteggi diretti
            accepted = sum(1 for yi, d in zip(c["y"], dec) if yi == 1 and d == 0)
            rejected = sum(1 for yi, d in zip(c["y"], dec) if yi == 0 and d == 1)
            self.check(name, f"apcer[{tag}]", apcer_exp, accepted / 4)
            self.check(name, f"bpcer[{tag}]", bpcer_exp, rejected / 4)
        try:
            dec_default = [int(v) for v in np.asarray(fn(s, 0.5)).ravel()]
            ok = dec_default == expected[False][0]
            REPORT_ROWS.append((name, "decisions[default]", str(expected[False][0]), str(dec_default), "PASS" if ok else "FAIL"))
            if not ok:
                self._failed.append("decisions[default]: default must be the strict rule s > t")
        except Exception as exc:  # noqa: BLE001
            self.note_error(name, "decisions[default]", f"{type(exc).__name__}: {exc}")
        self.finish(name)


class TestRoundTrip(MetricCase):
    """Punteggi scritti con repr in un CSV temporaneo, riletti: metriche identiche (uguaglianza esatta del dizionario)."""

    Y = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    S = [0.1 + 0.2, 1 / 3, 2 / 3, 0.5000001, 0.5, 0.49999999999999994,
         1e-17, 0.9999999999999999, 0.7, 0.12345678901234568, 0.5, 0.8]

    def _round_trip(self, name, y, s):
        # normalizzazione: liste di int e float, anche se arrivano array numpy
        y = [int(v) for v in list(y)]
        s = [float(v) for v in list(s)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["label", "score_attack"])
                for yi, si in zip(y, s):
                    w.writerow([int(yi), repr(float(si))])
            with open(path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        y2 = [int(r["label"]) for r in rows]
        s2 = [float(r["score_attack"]) for r in rows]
        ok_scores = y2 == y and s2 == s
        REPORT_ROWS.append((name, "scores_identical", "True", str(ok_scores), "PASS" if ok_scores else "FAIL"))
        if not ok_scores:
            self._failed.append("scores changed after the CSV round-trip")
        m1 = evaluation.compute_metrics(np.array(y, dtype=int), np.array(s, dtype=float))
        m2 = evaluation.compute_metrics(np.array(y2, dtype=int), np.array(s2, dtype=float))
        for key in sorted(set(m1) | set(m2)):
            if key == "roc":
                same = m1.get(key) == m2.get(key)
                REPORT_ROWS.append((name, key, "identical to in-memory", "identical" if same else "different",
                                    "PASS" if same else "FAIL"))
                if not same:
                    self._failed.append("roc differs")
                continue
            a, b = m1.get(key), m2.get(key)
            same = (a is None and b is None) or (a is not None and b is not None and a == b)
            REPORT_ROWS.append((name, key, fmt(a), fmt(b), "PASS" if same else "FAIL"))
            if not same:
                self._failed.append(f"{key}: in-memory {fmt(a)}, after round-trip {fmt(b)}")

    def test_15_round_trip_synthetic(self):
        name = "round_trip_synthetic"
        CASE_INFO[name] = f"repr -> CSV -> float, compute_metrics identical | y={self.Y} s={self.S}"
        self._round_trip(name, self.Y, self.S)
        self.finish(name)

    def test_16_round_trip_nuaa_minifasnet(self):
        name = "round_trip_nuaa_minifasnet"
        if not NUAA_CACHE.exists():
            if REQUIRE_RESTRICTED:
                self.note_error(name, "restricted_input", f"{rel(NUAA_CACHE)} not present (--require-restricted)")
                return self.finish(name)
            self.skipTest(f"{rel(NUAA_CACHE)} not present: real-data round-trip skipped")
        data, err = load_cache()
        if err or "minifasnet" not in data["scores"]:
            self.note_error(name, "cache", err or f"analyzer 'minifasnet' missing from {rel(NUAA_CACHE)}")
            return self.finish(name)
        y, s = load_cache_rows("minifasnet", data)
        CASE_INFO[name] = f"minifasnet scores from the NUAA cache (n={len(y)}), repr -> CSV -> float"
        self._round_trip(name, y, s)
        self.finish(name)


class TestAttackType(MetricCase):
    """metrics_by_attack_type: APCER per tipo con denominatore, BPCER globale, tipo atteso senza campioni."""

    def test_17_metrics_by_attack_type(self):
        name = "attack_types"
        y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        s = [0.1, 0.5, 0.6, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 0.51, 0.8, 0.3]
        types = ["", "", "", "", "", "print", "print", "print", "print", "replay", "replay", "replay"]
        CASE_INFO[name] = (f"y={y} s={s} attack_types={types} threshold=0.5 expected_types=['print','replay','mask']")
        fn = getattr(evaluation, "metrics_by_attack_type", None)
        if fn is None:
            self.note_error(name, "metrics_by_attack_type", "evaluation.metrics_by_attack_type is not available")
            return self.finish(name)
        try:
            r = fn(np.array(y), np.array(s), types, threshold=0.5, expected_types=["print", "replay", "mask"])
        except Exception as exc:  # noqa: BLE001
            self.note_error(name, "metrics_by_attack_type", f"{type(exc).__name__}: {exc}")
            return self.finish(name)
        self.check(name, "threshold", F(1, 2), r.get("threshold"))
        self.check(name, "n_bona_fide", 5, r.get("n_bona_fide"))
        self.check(name, "n_attack", 7, r.get("n_attack"))
        self.check(name, "rejected_bona_fide", 1, r.get("rejected_bona_fide"))
        self.check(name, "bpcer", F(1, 5), r.get("bpcer"))
        by_type = r.get("by_type") or {}
        keys_ok = set(by_type) == {"print", "replay", "mask"}
        REPORT_ROWS.append((name, "by_type.keys", "['mask', 'print', 'replay']", str(sorted(by_type)),
                            "PASS" if keys_ok else "FAIL"))
        if not keys_ok:
            self._failed.append(f"by_type keys: {sorted(by_type)}")
        expected = {"print": (4, 2, F(2, 4)), "replay": (3, 1, F(1, 3)), "mask": (0, 0, None)}
        for t, (n_att, accepted, apcer) in expected.items():
            d = by_type.get(t, {})
            self.check(name, f"{t}.n_attack", n_att, d.get("n_attack"))
            self.check(name, f"{t}.accepted", accepted, d.get("accepted"))
            self.check(name, f"{t}.apcer", apcer, d.get("apcer"))
        self.finish(name)

    def test_17b_attack_type_default_threshold(self):
        # stessi dati del caso default_threshold, senza l'argomento threshold: default 0,5, pareggio = bona fide
        name = "attack_types_default_threshold"
        y, s, types = [0, 0, 1, 1], [0.2, 0.495, 0.5, 0.8], ["", "", "print", "print"]
        CASE_INFO[name] = f"metrics_by_attack_type without threshold | y={y} s={s} attack_types={types}"
        try:
            r = evaluation.metrics_by_attack_type(np.array(y), np.array(s), types)
        except Exception as exc:  # noqa: BLE001
            self.note_error(name, "metrics_by_attack_type", f"{type(exc).__name__}: {exc}")
            return self.finish(name)
        self.check(name, "threshold", 0.5, r.get("threshold"))
        self.check(name, "bpcer", F(0, 2), r.get("bpcer"))
        self.check(name, "print.apcer", F(1, 2), (r.get("by_type") or {}).get("print", {}).get("apcer"))
        self.finish(name)


class TestInvalidInput(MetricCase):
    """Ingressi non validi: compute_metrics e metrics_by_attack_type devono sollevare ValueError (mai metriche calcolate
    su etichette troncate, punteggi NaN o tipi di attacco mancanti)."""

    def expect_value_error(self, case, what, fn):
        try:
            fn()
        except ValueError as exc:
            REPORT_ROWS.append((case, what, "ValueError", f"ValueError: {str(exc)[:40]}", "PASS"))
            return
        except Exception as exc:  # noqa: BLE001
            got = f"{type(exc).__name__}: {str(exc)[:30]}"
        else:
            got = "no exception"
        REPORT_ROWS.append((case, what, "ValueError", got, "FAIL"))
        self._failed.append(f"{what}: expected ValueError, obtained {got}")

    def test_20_compute_metrics_invalid(self):
        name = "invalid_compute_metrics"
        CASE_INFO[name] = "compute_metrics with labels outside {0, 1}, non-finite scores or mismatched lengths"
        cm = evaluation.compute_metrics
        s4 = [0.1, 0.4, 0.6, 0.9]
        self.expect_value_error(name, "label 2", lambda: cm(np.array([0, 1, 2, 1]), np.array(s4)))
        self.expect_value_error(name, "label -1", lambda: cm(np.array([0, -1, 1, 1]), np.array(s4)))
        self.expect_value_error(name, "label 0.7 (float)", lambda: cm(np.array([0, 0.7, 1, 1]), np.array(s4)))
        self.expect_value_error(name, "label 0.7 (list)", lambda: cm([0, 0.7, 1, 1], s4))
        self.expect_value_error(name, "label text", lambda: cm(np.array(["0", "0", "1", "1"]), np.array(s4)))
        self.expect_value_error(name, "score NaN", lambda: cm(np.array([0, 0, 1, 1]), np.array([0.1, np.nan, 0.6, 0.9])))
        self.expect_value_error(name, "score +inf", lambda: cm(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, np.inf, 0.9])))
        self.expect_value_error(name, "score -inf", lambda: cm(np.array([0, 0, 1, 1]), np.array([-np.inf, 0.4, 0.6, 0.9])))
        # una sola classe: né la AUC né la ROC vengono calcolate, quindi l'errore deve venire dal controllo del modulo
        self.expect_value_error(name, "score NaN, attacks only", lambda: cm(np.array([1, 1, 1]), np.array([0.2, np.nan, 0.9])))
        self.expect_value_error(name, "score inf, bona fide only", lambda: cm(np.array([0, 0]), np.array([np.inf, 0.1])))
        self.expect_value_error(name, "length mismatch", lambda: cm(np.array([0, 0, 1]), np.array(s4)))
        self.finish(name)

    def test_21_attack_type_invalid(self):
        name = "invalid_metrics_by_attack_type"
        CASE_INFO[name] = "metrics_by_attack_type with labels outside {0, 1}, NaN scores or a missing attack type"
        fn = evaluation.metrics_by_attack_type
        s4 = [0.1, 0.4, 0.6, 0.9]
        ok_types = ["", "", "print", "print"]
        self.expect_value_error(name, "label 2", lambda: fn(np.array([0, 0, 1, 2]), np.array(s4), ok_types))
        self.expect_value_error(name, "label -1", lambda: fn(np.array([0, -1, 1, 1]), np.array(s4), ok_types))
        self.expect_value_error(name, "label 0.7", lambda: fn(np.array([0, 0.7, 1, 1]), np.array(s4), ok_types))
        self.expect_value_error(name, "score NaN", lambda: fn(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, np.nan, 0.9]), ok_types))
        self.expect_value_error(name, "type NaN", lambda: fn(np.array([0, 0, 1, 1]), np.array(s4), ["", "", "print", float("nan")]))
        self.expect_value_error(name, "type numpy NaN", lambda: fn(np.array([0, 0, 1, 1]), np.array(s4), ["", "", np.float64("nan"), "print"]))
        self.expect_value_error(name, "type None", lambda: fn(np.array([0, 0, 1, 1]), np.array(s4), ["", "", "print", None]))
        self.expect_value_error(name, "type empty", lambda: fn(np.array([0, 0, 1, 1]), np.array(s4), ["", "", "", "print"]))
        self.expect_value_error(name, "type whitespace", lambda: fn(np.array([0, 0, 1, 1]), np.array(s4), ["", "", "print", "   "]))
        self.expect_value_error(name, "length mismatch", lambda: fn(np.array([0, 0, 1, 1]), np.array(s4), ["", "print"]))
        self.finish(name)


def rel(path):
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def load_cache():
    """Cache NUAA (sola lettura): (dati, None) oppure (None, messaggio) se il JSON non è valido o manca `scores`."""
    try:
        data = json.loads(NUAA_CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"{rel(NUAA_CACHE)} unreadable: {type(exc).__name__}: {exc}"
    if not isinstance(data, dict) or not isinstance(data.get("scores"), dict):
        return None, f"{rel(NUAA_CACHE)} has no 'scores' object"
    return data, None


def load_cache_rows(analyzer, data=None):
    """Etichette e punteggi di un analizzatore dalla cache NUAA (sola lettura), con lo stesso filtro di summarize():
    solo righe con punteggio finito, nell'ordine del file."""
    data = data if data is not None else load_cache()[0]
    per_img = data["scores"][analyzer]
    rows = [(int(v["y"]), float(v["s"])) for v in per_img.values()
            if isinstance(v.get("s"), (int, float)) and not isinstance(v.get("s"), bool) and math.isfinite(v["s"])]
    return [r[0] for r in rows], [r[1] for r in rows]


def finite_analyzers(data):
    """Analizzatori della cache con almeno un punteggio finito (le righe numeriche attese nella tabella)."""
    return {a for a, per in data["scores"].items() if isinstance(per, dict) and load_cache_rows(a, data)[0]}


def parse_c1_table():
    """Righe di results/c1/nuaa_table.md (sola lettura): (intestazione, righe numeriche [(analizzatore, {colonna: testo})],
    righe di nota [(analizzatore, testo)], righe sconosciute [testo]). Una riga numerica ha tutte le colonne; una nota ha
    tre celle con n = 0; ogni altra riga della tabella è sconosciuta."""
    lines = [ln for ln in NUAA_TABLE.read_text(encoding="utf-8").splitlines() if ln.startswith("|")]
    if len(lines) < 2:
        return [], [], [], ["table header not found"]
    header = [h.strip() for h in lines[0].strip("|").split("|")]
    numeric, notes, unknown = [], [], []
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) == len(header) and cells[1] != "0":
            numeric.append((cells[0], dict(zip(header, cells))))
        elif len(cells) == 3 and cells[1] == "0":
            notes.append((cells[0], cells[2]))
        else:
            unknown.append(ln[:60])
    return header, numeric, notes, unknown


class TestRealNuaa(MetricCase):
    """Righe reali della cache NUAA: attesi dall'implementazione di riferimento, confronto con il modulo (1e-12) e con
    la tabella C1 a 4 decimali. minifasnet è il caso richiesto; si controllano anche tutte le altre righe numeriche.
    Con la tabella presente, l'insieme delle righe numeriche deve essere esattamente quello degli analizzatori della
    cache con punteggi finiti (EXPECTED_REAL_ROWS righe, senza duplicati né righe sconosciute)."""

    def check_row_set(self, table_rows, notes, unknown, data):
        name = "real_nuaa:table_rows"
        CASE_INFO[name] = (f"numeric rows of {rel(NUAA_TABLE)} vs analyzers of {rel(NUAA_CACHE)} with finite scores "
                           f"(expected {EXPECTED_REAL_ROWS}, no duplicates, no unknown rows)")
        ids = [a for a, _ in table_rows]
        expected = finite_analyzers(data)
        dup = sorted(a for a in set(ids) if ids.count(a) > 1)
        dup_notes = sorted(a for a in set(n for n, _ in notes) if [x for x, _ in notes].count(a) > 1 or a in ids)
        checks = [
            ("rows_numeric", str(EXPECTED_REAL_ROWS), str(len(ids)), len(ids) == EXPECTED_REAL_ROWS),
            ("cache_with_scores", str(EXPECTED_REAL_ROWS), str(len(expected)), len(expected) == EXPECTED_REAL_ROWS),
            ("duplicated_rows", "[]", str(dup + dup_notes), not dup and not dup_notes),
            ("rows_not_in_cache", "[]", str(sorted(set(ids) - expected)), not set(ids) - expected),
            ("cache_without_row", "[]", str(sorted(expected - set(ids))), not expected - set(ids)),
            ("bad_note_rows", "[]",
             str(sorted(a for a, _ in notes if a not in data["scores"] or a in expected)),
             all(a in data["scores"] and a not in expected for a, _ in notes)),
            ("unknown_rows", "[]", str(unknown), not unknown),
        ]
        for metric, exp, got, ok in checks:
            REPORT_ROWS.append((name, metric, exp, got, "PASS" if ok else "FAIL"))
            if not ok:
                self._failed.append(f"{metric}: expected {exp}, obtained {got}")
        return list(self._failed)

    def test_18_real_nuaa(self):
        if not NUAA_CACHE.exists():
            if REQUIRE_RESTRICTED:
                self.note_error("real_nuaa", "restricted_input", f"{rel(NUAA_CACHE)} not present (--require-restricted)")
                return self.finish("real_nuaa")
            self.skipTest(f"{rel(NUAA_CACHE)} not present (exported package without cache): real case skipped")
        data, err = load_cache()
        if err:
            self.note_error("real_nuaa", "cache", err)
            return self.finish("real_nuaa")
        has_table = NUAA_TABLE.exists()
        failed_cases = []
        if has_table:
            _, numeric, notes, unknown = parse_c1_table()
            table = dict(numeric)
            failed_cases += [f"table rows ({f})" for f in self.check_row_set(numeric, notes, unknown, data)]
            # tutte le righe della tabella e tutti gli analizzatori con punteggi: un assente nella cache è un fallimento
            others = sorted(set(table) | finite_analyzers(data))
        else:
            table = {}
            # senza tabella: tutti gli analizzatori della cache con almeno due punteggi finiti
            others = [a for a in data["scores"] if isinstance(data["scores"][a], dict) and len(load_cache_rows(a, data)[0]) >= 2]
        analyzers = ["minifasnet"] + sorted(a for a in others if a != "minifasnet")
        for aid in analyzers:
            self._failed = []
            name = f"real_nuaa:{aid}"
            if aid not in data["scores"] or not isinstance(data["scores"][aid], dict):
                CASE_INFO[name] = f"results/eval/nuaa.json scores['{aid}']"
                self.note_error(name, "cache", f"analyzer '{aid}' missing from {rel(NUAA_CACHE)}")
                failed_cases.append(f"{aid} ({'; '.join(self._failed)})")
                continue
            y, s = load_cache_rows(aid, data)
            CASE_INFO[name] = f"results/eval/nuaa.json scores['{aid}'] (n={len(y)}), threshold 0.5; C1 table row at 4 decimals"
            exp = ref_metrics(y, s, 0.5)
            m = evaluation.compute_metrics(np.array(y, dtype=int), np.array(s, dtype=float))
            for key in METRICS:
                self.check(name, key, exp[key], obtained_value(m, key))
            row = table.get(aid)
            if not has_table:
                status = "FAIL" if REQUIRE_RESTRICTED else "SKIP"
                REPORT_ROWS.append((name, "c1_table", "-", f"{rel(NUAA_TABLE)} not present: table comparison skipped",
                                    status))
                if REQUIRE_RESTRICTED:
                    self._failed.append(f"{rel(NUAA_TABLE)} not present (--require-restricted)")
            elif row is None:
                REPORT_ROWS.append((name, "c1_table_row", "present", "absent", "FAIL"))
                self._failed.append("row missing from the C1 table")
            else:
                col = {"bpcer_at_apcer10": "bpcer_at_apcer10"}
                for key in ("n", "n_bona_fide", "n_attack", "apcer", "bpcer", "acer", "eer", "bpcer_at_apcer10",
                            "auc", "accuracy"):
                    v = exp[key]
                    want = "—" if v is None else (str(v) if isinstance(v, int) else f"{float(v):.4f}")
                    got = row.get(col.get(key, key), "<missing>")
                    ok = want == got
                    REPORT_ROWS.append((name, f"c1_table.{key}", want, got, "PASS" if ok else "FAIL"))
                    if not ok:
                        self._failed.append(f"c1_table.{key}: expected {want}, table {got}")
            if self._failed:
                failed_cases.append(f"{aid} ({'; '.join(self._failed)})")
        self._failed = failed_cases
        self.finish("real_nuaa")


class TestReferenceSelfCheck(MetricCase):
    """L'implementazione di riferimento riproduce esattamente i valori scritti a mano (anche l'EER interpolato):
    protegge il caso reale da un errore nel riferimento stesso. Non usa il modulo in prova."""

    def test_19_reference_self_check(self):
        name = "reference_self_check"
        CASE_INFO[name] = "ref_metrics / ref_eer_interpolated vs the hand-written Fractions of every synthetic case"
        for cname, c in CASES.items():
            ref = ref_metrics(c["y"], c["s"], 0.5 if c["threshold"] is None else c["threshold"])
            diff = [k for k, v in c["expected"].items() if ref[k] != v]
            if "eer_interpolated" in c:
                bona = [x for yi, x in zip(c["y"], c["s"]) if yi == 0]
                att = [x for yi, x in zip(c["y"], c["s"]) if yi == 1]
                if ref_eer_interpolated(bona, att) != c["eer_interpolated"]:
                    diff.append("eer_interpolated")
            REPORT_ROWS.append((name, cname, "hand values", "reproduced" if not diff else f"differs on {diff}",
                                "PASS" if not diff else "FAIL"))
            if diff:
                self._failed.append(f"{cname}: {diff}")
        self.finish(name)


# ----------------------------------------------------------------------------- rapporto

def build_report(result, n_run):
    out = io.StringIO()
    p = lambda *a: print(*a, file=out)  # noqa: E731
    p("# check_metrics report")
    p(f"module: {Path(evaluation.__file__).relative_to(ROOT)} | tolerance for floats: {ABS_TOL} (absolute)")
    p("")
    width = (34, 22, 34, 34)
    current = None
    for case, metric, exp, got, status in REPORT_ROWS:
        if case != current:
            current = case
            p(f"== {case}")
            if case in CASE_INFO:
                p(f"   {CASE_INFO[case]}")
            p(f"   {'metric':<{width[1]}} {'expected':<{width[2]}} {'obtained':<{width[3]}} result")
        p(f"   {metric:<{width[1]}} {exp:<{width[2]}} {got:<{width[3]}} {status}")
    p("")
    n_fail_rows = sum(1 for r in REPORT_ROWS if r[4] == "FAIL")
    n_skip_rows = sum(1 for r in REPORT_ROWS if r[4] == "SKIP")
    p(f"checks: {len(REPORT_ROWS)}, pass: {len(REPORT_ROWS) - n_fail_rows - n_skip_rows}, fail: {n_fail_rows}, "
      f"skipped: {n_skip_rows}")
    p(f"test cases run: {n_run}, failures: {len(result.failures)}, errors: {len(result.errors)}, skipped: {len(result.skipped)}")
    for test, _ in result.failures + result.errors:
        p(f"FAILED: {test.id().split('.')[-1]}")
    for test, reason in result.skipped:
        p(f"SKIPPED: {test.id().split('.')[-1]}: {reason}")
    # casi saltati (test interi) più confronti saltati dentro un caso (tabella C1 assente)
    n_skipped = len(result.skipped) + (1 if n_skip_rows else 0)
    if not result.wasSuccessful():
        p("RESULT: FAIL")
    elif n_skipped:
        p(f"RESULT: PASS ({n_skipped} cases skipped: restricted inputs not found)")
    else:
        p("RESULT: PASS")
    return out.getvalue()


def main():
    ap = argparse.ArgumentParser(description="Controlli delle metriche PAD con valori attesi indipendenti")
    ap.add_argument("--report", type=Path, help="scrive il rapporto testuale anche in questo file")
    ap.add_argument("--verbose", action="store_true", help="stampa anche il log di unittest con i traceback")
    ap.add_argument("--require-restricted", action="store_true",
                    help="cache e tabella C1 obbligatorie: se mancano il caso fallisce invece di essere saltato")
    args = ap.parse_args()
    global REQUIRE_RESTRICTED
    REQUIRE_RESTRICTED = args.require_restricted
    # ordine fisso: sintetici, regola inclusiva, round-trip, tipi di attacco, ingressi non validi, caso reale, verifica del riferimento
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite(loader.loadTestsFromTestCase(cls) for cls in (
        TestSynthetic, TestInclusiveDecision, TestRoundTrip, TestAttackType, TestInvalidInput, TestRealNuaa,
        TestReferenceSelfCheck))
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    report = build_report(result, result.testsRun)
    print(report, end="")
    if args.verbose:
        print(log.getvalue())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
