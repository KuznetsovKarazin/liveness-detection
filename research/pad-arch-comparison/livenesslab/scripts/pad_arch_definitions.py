"""
Definizioni operative del confronto tra reti PAD (LivenessNet e ResNet-18; incarico del coordinatore sulla scheda C2,
17/11/2026), come funzioni pure. In questa consegna non sono eseguite su dati reali: fissano le definizioni e sono
provate su dati sintetici (scripts/check_pad_arch_comparison.py).

  (a) one_class_run, one_class_counts   run che a soglia 0,5 assegnano tutto il dev a una classe
  (b) compare_condition                 differenza appaiata ResNet-18 − LivenessNet di ACER, APCER e BPCER per unità,
                                        pareggi esatti, media, conteggi, intervallo bootstrap per soggetto, per seed
  (c) sign_of, compare_conditions       segno di una condizione e cambio di segno fra due condizioni, con l'inversione
                                        della sola stima puntuale, il contrasto appaiato e le unità che cambiano segno
  (d) leave_one_subject_out             sensibilità del segno ai soggetti, scelta dell'autore nella forma del protocollo
                                        N1 del coordinatore (24/10/2026):
                                        media di ΔACER ricalcolata escludendo un soggetto di test alla volta; riportata
                                        accanto al segno, non entra nelle regole di (c)

Regola di decisione del progetto in un solo punto (attack_decisions): attacco se score > soglia, pareggio = bona fide
(nuaa_folds_run.DECISION_RULE). Soglia EER: nuaa_folds_run.eer_threshold sul dev, invariata (condition_threshold).
Bootstrap: la ricetta del progetto (nuaa_session_ablation_run.BOOT, boot_picks, interval: 1.000 ricampionamenti,
default_rng(0), percentili 2,5-97,5, minimo 5 soggetti), con la statistica e la regola di scarto di nuaa_bn_test
(media sulle unità delle differenze appaiate, stessi indici per le due reti; ricampionamento scartato se la metrica di
un'unità qualsiasi non è definita; ricampionamenti validi riportati).
Le letture non nominano cause. Valori fuori dominio (etichette diverse da 0/1, punteggi non finiti o fuori da [0, 1],
soglia fuori da [0, 1], frame diversi fra le due reti, unità duplicate, condizioni con unità diverse) -> Failure.
"""
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nuaa_folds_run as FR  # noqa: E402
import nuaa_session_ablation_run as SR  # noqa: E402
import pad_arch_models as AM  # noqa: E402

fail = FR.fail
MINUEND, SUBTRAHEND = "ResNet-18", "LivenessNet"            # Δ = metrica(ResNet-18) − metrica(LivenessNet)
PATIENCES = (4, 12)
BN_STATES = ("ordinary", "re-estimated")
THRESHOLD_RULES = ("fixed_0.5", "dev_eer")
CONDITIONS = tuple((p, b, t) for p in PATIENCES for b in BN_STATES for t in THRESHOLD_RULES)
METRICS = ("acer", "apcer", "bpcer")
SIGN_RESNET_LOWER, SIGN_LIVENESSNET_LOWER, SIGN_UNRESOLVED = "ResNet-18 lower", "LivenessNet lower", "unresolved"


# ============================================================================ dominio e regola di decisione

def labels_scores(y, s, what: str) -> tuple:
    """Etichette (0 bona fide, 1 attacco) e punteggi d'attacco finiti in [0, 1], stessa lunghezza; altrimenti Failure."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s):
        fail(f"{what}: labels and scores must be two lists of the same length")
    if len(y) and not np.all(np.isin(y, (0, 1))):
        fail(f"{what}: labels must be 0 (bona fide) or 1 (attack)")
    if not np.all(np.isfinite(s)) or np.any(s < 0) or np.any(s > 1):
        fail(f"{what}: scores must be finite and in [0, 1]")
    return y.astype(np.int64), s


def check_threshold(t, what: str) -> float:
    if isinstance(t, bool) or not isinstance(t, (int, float, np.floating)) or not math.isfinite(t) or not 0 <= t <= 1:
        fail(f"{what}: the threshold must be a finite number in [0, 1]")
    return float(t)


def attack_decisions(s: np.ndarray, threshold: float) -> np.ndarray:
    """La regola del progetto (FR.DECISION_RULE): attacco se score > soglia, pareggio = bona fide."""
    return s > threshold


def condition_threshold(rule: str, dev_y, dev_s) -> float:
    """Soglia di una condizione: 0,5 fissa, oppure la soglia EER di FR.eer_threshold stimata sul dev del fold con gli
    score del modello nello stesso stato della BN (Failure se una classe manca nel dev)."""
    if rule == "fixed_0.5":
        return float(FR.FIXED_THRESHOLD)
    if rule == "dev_eer":
        y, s = labels_scores(dev_y, dev_s, "dev")
        return float(FR.eer_threshold(y, s)["value"])
    fail(f"unknown threshold rule {rule!r}: only {', '.join(THRESHOLD_RULES)}")


# ============================================================================ (a) run a una sola classe

def one_class_run(dev_y, dev_s) -> dict:
    """(a) Un modello addestrato in uno stato della BN (ordinaria o ristimata), sul dev del suo fold, a soglia 0,5.
    Se una delle due classi manca tra le etichette del dev il run è «non valutabile» (evaluable False, contato a parte,
    fuori dal denominatore). Altrimenti, con la regola del progetto (attacco se score > 0,5, pareggio = bona fide), il
    run è «a una sola classe» se e solo se il numero di frame decisi attacco è 0 (tutto bona fide) oppure è uguale al
    numero di frame del dev (tutto attacco).
    Differenza dal criterio «degenere» già usato (nuaa_degenerate_trainings.py:402-405: epoca ripristinata 1 o 2 e
    accuratezza dev 0,5 entro 1e-6): quello guarda l'accuratezza della scheda, che su un dev bilanciato vale 0,5 anche
    con esattamente metà dei frame sbagliati in entrambe le classi, e l'epoca ripristinata; questo guarda solo le
    decisioni a 0,5 dello stato della BN valutato, a qualunque epoca e con qualunque bilanciamento."""
    y, s = labels_scores(dev_y, dev_s, "dev")
    n = len(y)
    if n == 0:
        fail("dev: no frame")
    n_att, n_bf = int(np.sum(y == 1)), int(np.sum(y == 0))
    k = int(np.sum(attack_decisions(s, FR.FIXED_THRESHOLD)))
    out = {"n": n, "n_attack_labels": n_att, "n_bona_fide_labels": n_bf, "n_decided_attack": k, "threshold": float(FR.FIXED_THRESHOLD)}
    if n_att == 0 or n_bf == 0:
        return {**out, "evaluable": False, "one_class": None, "class": None}
    cls = "bona fide" if k == 0 else "attack" if k == n else None
    return {**out, "evaluable": True, "one_class": cls is not None, "class": cls}


def one_class_counts(records) -> list:
    """Conteggio di (a) per rete × patience × stato della BN: k run a una sola classe su n valutabili, più i non
    valutabili a parte. records: {"network", "patience", "bn_state", "result": uscita di one_class_run}."""
    cells = {}
    for r in records:
        if r["network"] not in AM.NETWORKS or r["patience"] not in PATIENCES or r["bn_state"] not in BN_STATES:
            fail("one-class count: network, patience or BN state outside the plan")
        c = cells.setdefault((r["network"], r["patience"], r["bn_state"]),
                             {"one_class": 0, "evaluable": 0, "not_evaluable": 0, "all_bona_fide": 0, "all_attack": 0})
        res = r["result"]
        if not res["evaluable"]:
            c["not_evaluable"] += 1
            continue
        c["evaluable"] += 1
        if res["one_class"]:
            c["one_class"] += 1
            c["all_bona_fide" if res["class"] == "bona fide" else "all_attack"] += 1
    return [{"network": k[0], "patience": k[1], "bn_state": k[2], **v, "text": f"{v['one_class']} of {v['evaluable']} evaluable"}
            for k, v in sorted(cells.items(), key=lambda kv: (AM.NETWORKS.index(kv[0][0]), kv[0][1], BN_STATES.index(kv[0][2])))]


# ============================================================================ (b) confronto tra reti

def subject_counts(y, s, subjects, threshold: float) -> dict:
    """Conteggi per soggetto (soggetti in ordine crescente): attacchi, bona fide, attacchi accettati (decisi bona fide),
    bona fide respinti (decisi attacco), con la regola del progetto."""
    y, s = labels_scores(y, s, "test")
    subjects = np.asarray(subjects).astype(str)
    if len(subjects) != len(y):
        fail("test: one subject per frame is required")
    d = attack_decisions(s, threshold)
    subs = sorted(set(subjects.tolist()))
    out = {"subjects": subs}
    for k in ("na", "nb", "acc", "rej"):
        out[k] = np.zeros(len(subs), np.int64)
    for i, sub in enumerate(subs):
        m = subjects == sub
        out["na"][i] = int(np.sum(m & (y == 1)))
        out["nb"][i] = int(np.sum(m & (y == 0)))
        out["acc"][i] = int(np.sum(m & (y == 1) & ~d))
        out["rej"][i] = int(np.sum(m & (y == 0) & d))
    return out


def rates(na: int, nb: int, acc: int, rej: int) -> dict:
    """APCER, BPCER, ACER come frazioni esatte dei conteggi interi; None se la classe del denominatore manca."""
    apcer = Fraction(int(acc), int(na)) if na else None
    bpcer = Fraction(int(rej), int(nb)) if nb else None
    acer = (apcer + bpcer) / 2 if apcer is not None and bpcer is not None else None
    return {"apcer": apcer, "bpcer": bpcer, "acer": acer}


def weighted_rates(c: dict, w: np.ndarray) -> dict:
    """rates dei frame uniti dei soggetti con molteplicità w (w nell'ordine di c["subjects"])."""
    return rates(int(w @ c["na"]), int(w @ c["nb"]), int(w @ c["acc"]), int(w @ c["rej"]))


def diff(a, b):
    return None if a is None or b is None else a - b


def prepare_units(units) -> list:
    """Unità di una condizione: {"unit": (dataset, fold o split, seed), "networks": {rete: {"y", "s", "subject",
    "threshold"}}}. Le due reti devono avere gli stessi frame (stesse etichette e soggetti, nello stesso ordine);
    unità tutte diverse; due unità dello stesso split con gli stessi soggetti. Restituisce i conteggi per soggetto."""
    out, seen, split_subjects = [], set(), {}
    if not units:
        fail("comparison: no unit")
    for u in units:
        key = tuple(u["unit"])
        if len(key) != 3 or key in seen:
            fail(f"comparison: unit {key} malformed or duplicated")
        seen.add(key)
        nets = u["networks"]
        if sorted(nets) != sorted(AM.NETWORKS):
            fail(f"comparison: unit {key} must hold exactly {', '.join(AM.NETWORKS)}")
        a, b = nets[MINUEND], nets[SUBTRAHEND]
        if len(a["y"]) != len(b["y"]) or not np.array_equal(np.asarray(a["y"]), np.asarray(b["y"])) \
                or not np.array_equal(np.asarray(a["subject"]).astype(str), np.asarray(b["subject"]).astype(str)):
            fail(f"comparison: unit {key}: the two networks are not scored on the same test frames")
        counts = {n: subject_counts(nets[n]["y"], nets[n]["s"], nets[n]["subject"], check_threshold(nets[n]["threshold"], f"unit {key} {n}"))
                  for n in AM.NETWORKS}
        if not counts[MINUEND]["subjects"]:
            fail(f"comparison: unit {key} has no test frame")
        split = (key[0], key[1])
        if split_subjects.setdefault(split, counts[MINUEND]["subjects"]) != counts[MINUEND]["subjects"]:
            fail(f"comparison: two units of {split} are not scored on the same subjects")
        out.append({"unit": key, "split": split, "seed": key[2], "counts": counts})
    return out


def unit_row(pu: dict) -> dict:
    """Riga di un'unità: metriche esatte delle due reti, Δ (ResNet-18 − LivenessNet) e pareggio di ACER come frazioni."""
    ones = {n: np.ones(len(pu["counts"][n]["subjects"]), np.int64) for n in AM.NETWORKS}
    m = {n: weighted_rates(pu["counts"][n], ones[n]) for n in AM.NETWORKS}
    d = {k: diff(m[MINUEND][k], m[SUBTRAHEND][k]) for k in METRICS}
    return {"unit": pu["unit"], "metrics": m, "delta": d, "tie": None if d["acer"] is None else d["acer"] == 0}


def pooled_index(pus: list) -> tuple:
    """Soggetti di test uniti sugli split (ordine crescente) e, per split, gli indici dei suoi soggetti; un soggetto in
    due split è rifiutato."""
    subj = {}
    for pu in pus:
        subj.setdefault(pu["split"], pu["counts"][MINUEND]["subjects"])
    pooled = sorted({s for v in subj.values() for s in v})
    if len(pooled) != sum(len(v) for v in subj.values()):
        fail("comparison: a test subject appears in two splits")
    pos = {s: i for i, s in enumerate(pooled)}
    return pooled, {sp: np.array([pos[s] for s in v], np.int64) for sp, v in subj.items()}


def subject_bootstrap(pus: list, value) -> dict:
    """Intervallo per soggetto della ricetta del progetto (SR.boot_picks, SR.interval) sui soggetti uniti; a ogni
    ricampionamento ogni unità usa le molteplicità dei soggetti del proprio split, le stesse per le due reti (e per le
    due condizioni di un contrasto); value(pu, wf) -> {metrica: valore o None}; statistica = media sulle unità; un
    ricampionamento in cui la metrica di un'unità qualsiasi non è definita non conta per quella metrica."""
    pooled, index = pooled_index(pus)
    n = len(pooled)
    out = {"n_subjects": n, "resamples": SR.BOOT["resamples"], "seed": SR.BOOT["seed"], "percentiles": list(SR.BOOT["percentiles"])}
    if n < SR.BOOT["min_subjects"]:
        return {**out, "note": f"no interval: fewer than {SR.BOOT['min_subjects']} subjects",
                **{k: {"low": None, "high": None, "valid_resamples": 0} for k in METRICS}}
    vals = {k: [] for k in METRICS}
    for pick in SR.boot_picks(n):
        w = np.bincount(pick, minlength=n)
        per = [value(pu, w[index[pu["split"]]]) for pu in pus]
        for k in METRICS:
            if all(v[k] is not None for v in per):
                vals[k].append(float(np.mean([float(v[k]) for v in per])))
    return {**out, **{k: SR.interval(vals[k]) for k in METRICS}}


def delta_at(pu: dict, wf: np.ndarray) -> dict:
    a, b = weighted_rates(pu["counts"][MINUEND], wf), weighted_rates(pu["counts"][SUBTRAHEND], wf)
    return {k: diff(a[k], b[k]) for k in METRICS}


def summarise(pus: list) -> dict:
    """Riepilogo di un insieme di unità: media di Δ per metrica (None se Δ non è definita in un'unità), conteggi delle
    unità con ΔACER < 0, > 0, pareggio (frazioni esatte) e non definita, intervallo bootstrap per soggetto."""
    rows = [unit_row(pu) for pu in pus]
    mean = {}
    for k in METRICS:
        d = [r["delta"][k] for r in rows]
        mean[k] = float(sum(d) / len(d)) if all(x is not None for x in d) else None
    da = [r["delta"]["acer"] for r in rows]
    counts = {"resnet18_lower": sum(1 for x in da if x is not None and x < 0), "livenessnet_lower": sum(1 for x in da if x is not None and x > 0),
              "ties": sum(1 for x in da if x is not None and x == 0), "undefined": sum(1 for x in da if x is None), "units": len(rows)}
    boot = subject_bootstrap(pus, delta_at)
    return {"mean_delta": mean, "acer_counts": counts, "bootstrap": boot,
            "interval": {k: {"low": boot[k]["low"], "high": boot[k]["high"], "valid_resamples": boot[k]["valid_resamples"]} for k in METRICS},
            "rows": rows}


def compare_condition(units) -> dict:
    """(b) Una condizione (patience, stato BN, soglia): per unità Δ = metrica(ResNet-18) − metrica(LivenessNet) per
    ACER, APCER e BPCER sugli stessi frame di test; pareggio di un'unità = i due ACER uguali come frazioni esatte dei
    conteggi interi. Riepilogo su tutte le unità e, a parte, per ciascun seed: media di Δ, conteggi Δ < 0, Δ > 0 e
    pareggi, intervallo bootstrap per soggetto (soggetti di test uniti sui fold, stessi indici per le due reti, media
    sulle unità delle differenze appaiate, ricampionamento scartato se una metrica non è definita, ricampionamenti
    validi riportati). Il segno della condizione è sign_of(intervallo di ACER)."""
    pus = prepare_units(units)
    out = summarise(pus)
    out["sign"] = sign_of(out["interval"]["acer"])
    out["per_seed"] = {}
    for seed in sorted({pu["seed"] for pu in pus}):
        sub = summarise([pu for pu in pus if pu["seed"] == seed])
        sub["sign"] = sign_of(sub["interval"]["acer"])
        out["per_seed"][seed] = sub
    out["prepared"] = pus
    return out


# ============================================================================ (c) segno e cambio di segno

def sign_of(interval: dict) -> str:
    """(c) Segno di una condizione dall'intervallo di ΔACER: «ResNet-18 lower» se è tutto sotto zero, «LivenessNet
    lower» se è tutto sopra, «unresolved» se contiene lo zero (un estremo esattamente a zero lo contiene) o non è
    calcolabile."""
    lo, hi = interval.get("low"), interval.get("high")
    if lo is None or hi is None:
        return SIGN_UNRESOLVED
    if hi < 0:
        return SIGN_RESNET_LOWER
    if lo > 0:
        return SIGN_LIVENESSNET_LOWER
    return SIGN_UNRESOLVED


def compare_conditions(a: dict, b: dict) -> dict:
    """(c) Due condizioni già confrontate con compare_condition, sulle stesse unità e sugli stessi frame. Cambio di segno
    riconosciuto solo se entrambe sono risolte e di segno opposto. A parte, con nomi diversi: l'inversione della sola
    stima puntuale (medie di ΔACER di segno opposto, entrambe diverse da zero, con almeno una condizione non risolta);
    il contrasto appaiato Δ_a − Δ_b (media sulle unità e intervallo con gli stessi indici di ricampionamento per le due
    condizioni e le due reti); il numero di unità il cui ΔACER ha segno strettamente opposto nelle due condizioni.
    Nessuna lettura nomina cause."""
    pa, pb = a["prepared"], b["prepared"]
    if [p["unit"] for p in pa] != [p["unit"] for p in pb]:
        fail("sign change: the two conditions are not on the same units")
    for x, y in zip(pa, pb):
        cx, cy = x["counts"][MINUEND], y["counts"][MINUEND]
        if cx["subjects"] != cy["subjects"] or any(not np.array_equal(cx[k], cy[k]) for k in ("na", "nb")):
            fail(f"sign change: unit {x['unit']} is not scored on the same test frames in the two conditions")
    sa, sb = a["sign"], b["sign"]
    resolved = sa != SIGN_UNRESOLVED and sb != SIGN_UNRESOLVED
    ma, mb = a["mean_delta"]["acer"], b["mean_delta"]["acer"]
    opposite_means = ma is not None and mb is not None and ma * mb < 0
    pair = {x["unit"]: (x, y) for x, y in zip(pa, pb)}
    rows = []
    for x, y in zip(pa, pb):
        ones = np.ones(len(x["counts"][MINUEND]["subjects"]), np.int64)
        da, db = delta_at(x, ones), delta_at(y, ones)
        rows.append({"unit": x["unit"], "delta_a": da["acer"], "delta_b": db["acer"],
                     "contrast": {k: diff(da[k], db[k]) for k in METRICS}})

    def contrast_at(pu, wf):
        x, y = pair[pu["unit"]]
        da, db = delta_at(x, wf), delta_at(y, wf)
        return {k: diff(da[k], db[k]) for k in METRICS}
    boot = subject_bootstrap(pa, contrast_at)
    cmean = {k: (float(sum(r["contrast"][k] for r in rows) / len(rows)) if all(r["contrast"][k] is not None for r in rows) else None)
             for k in METRICS}
    changed = sum(1 for r in rows if r["delta_a"] is not None and r["delta_b"] is not None and r["delta_a"] * r["delta_b"] < 0)
    return {"sign_a": sa, "sign_b": sb, "sign_change": bool(resolved and sa != sb),
            "point_estimate_inversion": bool(opposite_means and not resolved),
            "contrast_mean": cmean, "contrast_interval": {k: {"low": boot[k]["low"], "high": boot[k]["high"],
                                                              "valid_resamples": boot[k]["valid_resamples"]} for k in METRICS},
            "units_changing_sign": changed, "units": len(rows)}


# ============================================================================ (d) sensibilità ai soggetti

def sign_value(v) -> int:
    return (v > 0) - (v < 0)


def leave_one_subject_out(units) -> dict:
    """(d) Sensibilità del segno ai soggetti: analisi scelta dall'autore (non richiesta nell'incarico), nella forma della
    definizione del protocollo N1 del coordinatore (24/10/2026; scheda N1, consegna N1, PR #10): per una
    condizione, la media sulle unità di ΔACER = ACER(ResNet-18) − ACER(LivenessNet) ricalcolata escludendo dal solo
    calcolo delle metriche un soggetto di test alla volta, su tutte le unità (il soggetto esce dalle unità del suo
    split; le altre restano intere). Per ogni soggetto escluso: il valore, il suo segno, se il segno differisce da quello
    della stima con tutti i soggetti (segni -1, 0, +1 delle frazioni esatte: Δ = 0 è un segno a sé) e i denominatori per
    classe di ogni unità. Se dopo l'esclusione una classe manca in un'unità il valore è NA (None), mai zero, ed è contato
    a parte (not_available). Minimo e massimo sui valori definiti. Analisi di sensibilità riportata accanto al segno:
    non entra in sign_of né in compare_conditions."""
    pus = prepare_units(units)
    pooled, _ = pooled_index(pus)

    def mean_acer(excluded):
        vals, dens = [], []
        for pu in pus:
            keep = np.array([s != excluded for s in pu["counts"][MINUEND]["subjects"]], np.int64)
            c = pu["counts"][MINUEND]
            na, nb = int(keep @ c["na"]), int(keep @ c["nb"])
            dens.append({"unit": pu["unit"], "n_attack": na, "n_bona_fide": nb})
            vals.append(delta_at(pu, keep)["acer"])
        return (None if any(v is None for v in vals) else sum(vals) / len(vals)), dens
    full, _ = mean_acer(None)
    full_sign = None if full is None else sign_value(full)
    values = []
    for s in pooled:
        v, dens = mean_acer(s)
        sg = None if v is None else sign_value(v)
        values.append({"excluded_subject": s, "mean_delta_acer": None if v is None else float(v), "sign": sg,
                       "sign_differs": None if v is None or full_sign is None else sg != full_sign, "denominators": dens})
    defined = [x["mean_delta_acer"] for x in values if x["mean_delta_acer"] is not None]
    return {"full_mean_delta_acer": None if full is None else float(full), "full_sign": full_sign, "values": values,
            "min": min(defined) if defined else None, "max": max(defined) if defined else None,
            "n_sign_differs": sum(1 for x in values if x["sign_differs"]),
            "n_not_available": sum(1 for x in values if x["mean_delta_acer"] is None), "n_excluded": len(values)}
