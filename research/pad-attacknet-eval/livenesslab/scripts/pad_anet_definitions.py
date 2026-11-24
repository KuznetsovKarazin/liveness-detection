"""
Definizioni operative del confronto C5 fra le cinque reti PAD su NUAA (LivenessNet, AttackNetV1, AttackNetV2_1,
AttackNetV2_2 e ResNet-18 come riferimento esterno; scheda C5 del coordinatore, 21/11/2026), come funzioni pure. In
questa consegna non sono eseguite su dati reali: fissano le definizioni prima di qualunque nuovo risultato di test e
sono provate su dati sintetici (scripts/check_pad_anet_comparison.py), anche contro il codice congelato di C2
(pad_arch_definitions.primary_analysis, confronto incrociato solo interno).

  (a) variazioni e contrasti per unità, in frazioni esatte (quantity, primary_analysis): per un'unità u e una rete n,
      v_n(u) = ACER_n,ristimata(u) − ACER_n,ordinaria(u) come Fraction; contrasto c_k(u) = (A_k,re − A_k,ord) −
      (L_re − L_ord) come Fraction, mai come differenza di float; V_n e C_k = media Fraction sulle 12 unità, in float
      solo all'uscita; nel bootstrap, a ogni ricampionamento, il valore per unità dai conteggi pesati
      (pad_arch_definitions.weighted_rates) in frazioni e la media dei float delle frazioni per unità (come
      pad_arch_definitions.compare_conditions); lo stesso per APCER e BPCER
  (b) intervalli individuali: nessuna correzione per molteplicità, nessuna copertura simultanea (INTERVALS_NOTE)
  (c) lettura di ciascun contrasto (contrast_reading) e conclusioni scritte prima (conclusion), una per AttackNet;
      nessuna affermazione congiunta sui tre contrasti
  (d) risultati per seed a parte (per_seed di primary_analysis), fuori dalla lettura
  (e) ID dei frame: per unità le reti presenti sugli stessi ID nello stesso ordine (prepare), fra i due stati della BN
      (pad_arch_definitions.same_frames)
  (f) classe unica: pad_arch_definitions.one_class_run sul dev (one_class_counts per le cinque reti); sul test, conteggio
      descrittivo dei modelli che decidono tutto una classe (test_one_class)
  (g) graduatorie descrittive (rankings): per condizione le reti ordinate per ACER medio in frazioni esatte, pari merito
      con lo stesso rango minimo; per unità i conteggi dei ranghi; nessun intervallo
  (h) denominatori (denominators): unità, soggetti, frame di test sommati sulle unità, attacchi e bona fide per fold e
      in totale, ricampionamenti validi
  (i) punti percentuali solo in presentazione (pp, pp_text): proporzione × 100, due decimali
  (k)/(n) training non finito e modello non valutabile (finite_argmin, history_outcome, training_outcome, dev_outcome),
      guardiano per rete (guardian)
  (m) confronto descrittivo con ResNet-18 (primary_analysis, descriptive_vs_resnet18): per ognuna delle quattro reti
      nostre, per stato BN il divario metrica(n) − metrica(ResNet-18) e V_n − V_R; nessuna lettura

Regola di decisione, soglie, bootstrap e dominio dei dati sono quelli di C2, importati così come sono
(pad_arch_definitions: labels_scores, check_threshold, attack_decisions, condition_threshold, subject_counts, rates,
weighted_rates, diff, frame_ids, ids_sha256, same_frames, finite_real, bounds, interval_reason, one_class_run;
nuaa_session_ablation_run: boot_picks, interval, BOOT). Le funzioni di C2 che contengono le due reti (prepare_units,
pooled_index, unit_row, delta_at, summarise, compare_condition, compare_conditions, primary_analysis, sign_of,
contrast_reading, one_class_counts) sono sostituite qui da funzioni con le reti come parametri; nessuna riga di
pad_arch_definitions cambia. Nessun messaggio d'errore e nessuna uscita riepilogativa contiene ID di frame o di
soggetto; il campo interno «prepared» e le righe per unità con le frazioni non sono uscite pubblicabili.
"""
import math
import sys
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nuaa_folds_run as FR  # noqa: E402
import nuaa_session_ablation_run as SR  # noqa: E402
import pad_anet_models as AN  # noqa: E402
import pad_arch_definitions as D  # noqa: E402
import pad_arch_models as AM  # noqa: E402

fail = D.fail
ATTACKNETS = AN.NETWORKS
REFERENCE, EXTERNAL = "LivenessNet", "ResNet-18"
REUSED = AM.NETWORKS                                      # ("LivenessNet", "ResNet-18"): i 48 riferimenti di C2
OURS = (REFERENCE,) + ATTACKNETS                          # le «quattro reti nostre»
NETWORKS = OURS + (EXTERNAL,)
PATIENCES, BN_STATES, THRESHOLD_RULES, CONDITIONS, METRICS = D.PATIENCES, D.BN_STATES, D.THRESHOLD_RULES, D.CONDITIONS, D.METRICS
PRIMARY_PATIENCE, PRIMARY_THRESHOLD = D.PRIMARY_PATIENCE, D.PRIMARY_THRESHOLD
PRIMARY_ORDINARY, PRIMARY_RE_ESTIMATED = D.PRIMARY_ORDINARY, D.PRIMARY_RE_ESTIMATED
PLAN_SEEDS = D.PLAN_SEEDS
FOLDS = (1, 2, 3, 4)
UNITS_REQUIRED = len(FOLDS) * len(PLAN_SEEDS)             # 12 coppie fold-seed
FIELDS = D.NETWORK_FIELDS
HISTORY_KEYS = ("loss", "accuracy", "val_loss", "val_accuracy", "lr")
INTERVALS_NOTE = ("individual 95% intervals: no correction for multiplicity, no simultaneous coverage over the contrasts, "
                  "variations or comparisons")
PP_NOTE = ("percentage points = proportion x 100, rounded to two decimals half away from zero from the exact value, at presentation only; "
           "the computations use exact fractions")
UNRESOLVED = "unresolved"
NOT_COMPUTED = "not computed"
NOT_EVALUABLE = "not evaluable"


def reading_names(k: str) -> tuple:
    """I tre nomi della lettura del contrasto di una AttackNet (contrast_reading)."""
    return (f"ACER variation of {k} minus that of LivenessNet lower than zero",
            f"ACER variation of {k} minus that of LivenessNet higher than zero", UNRESOLVED)


# ============================================================================ (k)/(n) training non finito, modello non valutabile

def finite_or_none(v):
    """Un valore della storia: float se finito, None altrimenti (NaN, infinito): la forma JSON della regola (n)."""
    return float(v) if D.finite_real(v) else None


def finite_argmin(values):
    """L'indice (0-based) della val_loss finita minima, il primo a pari valore; None se nessun valore è finito. Al posto di
    np.argmin, che con un NaN restituisce il primo NaN."""
    best = None
    for i, v in enumerate(values):
        if D.finite_real(v) and (best is None or float(v) < float(values[best])):
            best = i
    return best


def non_finite_entries(history: dict) -> list:
    """I valori non finiti di una storia: [grandezza, prima epoca (1-based)] per ogni grandezza di HISTORY_KEYS presente
    che ne ha uno."""
    out = []
    for k in HISTORY_KEYS:
        for i, v in enumerate(history.get(k, [])):
            if not D.finite_real(v):
                out.append([k, i + 1])
                break
    return out


def history_outcome(history: dict) -> dict:
    """Regola (n), prima parte: la storia di un addestramento (valori anche non finiti). Un valore non finito non ferma il
    run: si registrano grandezza e prima epoca non finita; l'epoca ripristinata è quella della val_loss finita minima
    (finite_argmin); senza alcuna val_loss finita il modello è non valutabile («no finite val_loss»)."""
    if not isinstance(history, dict) or "val_loss" not in history:
        fail("history: val_loss is required")
    best = finite_argmin(history["val_loss"])
    return {"non_finite": non_finite_entries(history), "restored_epoch": None if best is None else best + 1,
            "evaluable": best is not None, "reason": None if best is not None else f"{NOT_EVALUABLE}: no finite val_loss"}


def training_outcome(history: dict, weights_finite) -> dict:
    """Regola (n): esito di un addestramento dopo il ripristino dei pesi. Valutabile solo se esiste una val_loss finita
    e tutti i tensori dei pesi ripristinati sono finiti (weights_finite True); altrimenti «not evaluable: <motivo>»:
    niente checkpoint, niente fase 2, niente test. weights_finite None è ammesso solo quando non c'è val_loss finita."""
    h = history_outcome(history)
    if not h["evaluable"]:
        return h
    if weights_finite is not True and weights_finite is not False:
        fail("training outcome: weights_finite must be True or False when a finite val_loss exists")
    if not weights_finite:
        return {**h, "evaluable": False, "reason": f"{NOT_EVALUABLE}: restored weights not finite"}
    return h


def arrays_finite(arrays) -> bool:
    """Tutti i valori di tutti gli array finiti (pesi, punteggi, logit, statistiche)."""
    return all(bool(np.all(np.isfinite(np.asarray(a, dtype=np.float64)))) for a in arrays)


def dev_outcome(states: dict, statistics_finite: bool) -> dict:
    """Regola (n), seconda parte, prima della ristima e prima della scrittura delle uscite del modello: non valutabile se
    il dev, in uno dei due stati, ha punteggi o logit non finiti (states: stato -> {"p", "z"}, None per uno stato non
    calcolato) o se la ristima dà statistiche non finite."""
    for s in BN_STATES:
        st = states.get(s)
        if st is not None and not arrays_finite([st["p"], st["z"]]):
            return {"evaluable": False, "reason": f"{NOT_EVALUABLE}: non-finite dev scores or logits ({s} BN)"}
    if statistics_finite is not True:
        return {"evaluable": False, "reason": f"{NOT_EVALUABLE}: non-finite re-estimated statistics"}
    return {"evaluable": True, "reason": None}


# ============================================================================ (e) unità di una condizione

def prepare(units, required=()) -> list:
    """Unità di una condizione: {"unit": (dataset, fold, seed), "networks": {rete: {"y", "s", "subject", "threshold",
    "sample_id"}}} con le reti fra NETWORKS (un'AttackNet non valutabile manca), le reti required presenti in ogni unità,
    esattamente quei campi. Unico punto dei controlli sui frame di una condizione (le regole di
    pad_arch_definitions.prepare_units, per un insieme di reti qualunque):
      - le reti presenti in un'unità hanno le stesse etichette e gli stessi soggetti nello stesso ordine e, controllo a
        parte, gli stessi ID dei frame nello stesso ordine (D.frame_ids);
      - unità tutte diverse; seed intero fra PLAN_SEEDS; un solo dataset;
      - due unità dello stesso fold hanno la stessa sequenza di (ID, soggetto, etichetta); un ID in un solo fold.
    Restituisce per unità le reti presenti, i conteggi per soggetto di ciascuna (D.subject_counts), la soglia effettiva,
    i frame (campo interno), il numero dei frame e l'impronta degli ID."""
    out, seen, split_frames, id_split = [], set(), {}, {}
    if not units:
        fail("comparison: no unit")
    for u in units:
        key = tuple(u["unit"])
        if len(key) != 3 or key in seen:
            fail(f"comparison: unit {key} malformed or duplicated")
        if isinstance(key[2], (bool, np.bool_)) or not isinstance(key[2], (int, np.integer)) or int(key[2]) not in PLAN_SEEDS:
            fail(f"comparison: unit {key}: the seed must be an integer among the seeds of the plan {list(PLAN_SEEDS)}")
        if key[0] != tuple(units[0]["unit"])[0]:
            fail("comparison: the units of one call must belong to one dataset")
        seen.add(key)
        nets = u["networks"]
        if not isinstance(nets, dict) or not nets or any(n not in NETWORKS for n in nets):
            fail(f"comparison: unit {key}: networks must be among {', '.join(NETWORKS)}")
        missing = [n for n in required if n not in nets]
        if missing:
            fail(f"comparison: unit {key} lacks {', '.join(missing)}")
        present = tuple(n for n in NETWORKS if n in nets)
        for n in present:
            if not isinstance(nets[n], dict) or sorted(nets[n]) != sorted(FIELDS):
                fail(f"comparison: unit {key} {n} must hold exactly the fields {', '.join(FIELDS)}")
        first = nets[present[0]]
        for n in present[1:]:
            b = nets[n]
            if len(first["y"]) != len(b["y"]) or not np.array_equal(np.asarray(first["y"]), np.asarray(b["y"])) \
                    or not np.array_equal(np.asarray(first["subject"]).astype(str), np.asarray(b["subject"]).astype(str)):
                fail(f"comparison: unit {key}: {present[0]} and {n} are not scored on the same test frames")
        thr = {n: D.check_threshold(nets[n]["threshold"], f"unit {key} {n}") for n in present}
        counts = {n: D.subject_counts(nets[n]["y"], nets[n]["s"], nets[n]["subject"], thr[n]) for n in present}
        if not counts[present[0]]["subjects"]:
            fail(f"comparison: unit {key} has no test frame")
        ids = {n: D.frame_ids(nets[n][D.FRAME_ID_FIELD], len(nets[n]["y"]), f"comparison: unit {key} {n}") for n in present}
        for n in present[1:]:
            if ids[n] != ids[present[0]]:
                fail(f"comparison: unit {key}: {present[0]} and {n} are not scored on the same frame identifiers in the same order")
        frames = (ids[present[0]], tuple(np.asarray(first["subject"]).astype(str).tolist()), tuple(int(x) for x in np.asarray(first["y"])))
        split = (key[0], key[1])
        if split not in split_frames:
            for i in frames[0]:
                if id_split.setdefault(i, split) != split:
                    fail(f"comparison: a frame identifier of {split} also appears in {id_split[i]}")
        if split_frames.setdefault(split, frames) != frames:
            fail(f"comparison: two units of {split} are not scored on the same frames (identifiers, subjects and labels in the same order)")
        out.append({"unit": key, "split": split, "seed": int(key[2]), "networks": present, "counts": counts, "thresholds": thr,
                    "frames": frames, "n_frames": len(frames[0]), "frame_ids_sha256": D.ids_sha256(frames[0])})
    return out


def pooled(pus: list) -> tuple:
    """Soggetti di test uniti sui fold (ordine crescente) e, per fold, gli indici dei suoi soggetti (la regola di
    pad_arch_definitions.pooled_index, con i soggetti della prima rete presente); un soggetto in due fold è rifiutato."""
    subj = {}
    for pu in pus:
        subj.setdefault(pu["split"], pu["counts"][pu["networks"][0]]["subjects"])
    allsub = sorted({s for v in subj.values() for s in v})
    if len(allsub) != sum(len(v) for v in subj.values()):
        fail("comparison: a test subject appears in two folds")
    pos = {s: i for i, s in enumerate(allsub)}
    return allsub, {sp: np.array([pos[s] for s in v], np.int64) for sp, v in subj.items()}


def pair_states(ordinary: list, re_estimated: list) -> list:
    """Le unità preparate dei due stati della BN appaiate per unità, nello stesso ordine e sugli stessi frame
    (D.same_frames: soggetti ed etichette, poi ID); stato -> unità preparata."""
    if [p["unit"] for p in ordinary] != [p["unit"] for p in re_estimated]:
        fail("comparison: the two BN states are not on the same units in the same order")
    keys = [(p["unit"][1], p["unit"][2]) for p in ordinary]
    if keys != sorted(keys):
        fail("comparison: the units must be in ascending (fold, seed) order (the order of the C2 units, on which the float means depend)")
    for x, y in zip(ordinary, re_estimated):
        D.same_frames(x, y, "the two BN states")
    return [{"ordinary": x, "re-estimated": y} for x, y in zip(ordinary, re_estimated)]


# ============================================================================ (a) quantità lineari per unità e bootstrap

def _rates(pu: dict, net: str, w=None) -> dict:
    c = pu["counts"][net]
    return D.weighted_rates(c, np.ones(len(c["subjects"]), np.int64) if w is None else w)


def term_value(entry: dict, terms, w=None, cache=None) -> dict:
    """Il valore per unità di una quantità lineare: somma di coefficiente × metrica(rete, stato) per ACER, APCER e BPCER,
    in frazioni esatte; None se una metrica non è definita o una rete manca. terms: ((coefficiente intero, rete, stato),
    ...); entry: stato -> unità preparata."""
    out = {}
    rs = {}
    for coef, net, state in terms:
        pu = entry[state]
        if net not in pu["counts"]:
            return {k: None for k in METRICS}
        key = (pu["unit"], net, state)
        if cache is not None and key in cache:
            rs[(net, state)] = cache[key]
        else:
            rs[(net, state)] = _rates(pu, net, w)
            if cache is not None:
                cache[key] = rs[(net, state)]
    for k in METRICS:
        vals = [rs[(net, state)][k] for _, net, state in terms]
        out[k] = None if any(v is None for v in vals) else sum((Fraction(int(c)) * v for (c, _, _), v in zip(terms, vals)), Fraction(0))
    return out


def variation_terms(net: str) -> tuple:
    """v_n = metrica(n, ristimata) − metrica(n, ordinaria)."""
    return ((1, net, "re-estimated"), (-1, net, "ordinary"))


def contrast_terms(k: str) -> tuple:
    """c_k = (A_k,re − A_k,ord) − (L_re − L_ord)."""
    return ((1, k, "re-estimated"), (-1, k, "ordinary"), (-1, REFERENCE, "re-estimated"), (1, REFERENCE, "ordinary"))


def gap_terms(net: str, state: str) -> tuple:
    """Divario descrittivo metrica(n) − metrica(ResNet-18) nello stato dato."""
    return ((1, net, state), (-1, EXTERNAL, state))


def variation_difference_terms(net: str) -> tuple:
    """V_n − V_R per unità: (n_re − n_ord) − (R_re − R_ord)."""
    return ((1, net, "re-estimated"), (-1, net, "ordinary"), (-1, EXTERNAL, "re-estimated"), (1, EXTERNAL, "ordinary"))


def exact_mean(values: list):
    """La media Fraction sulle unità (None se un valore manca)."""
    return None if not values or any(v is None for v in values) else sum(values, Fraction(0)) / len(values)


def quantities(entries: list, specs: dict) -> dict:
    """Più quantità lineari sulle stesse unità appaiate, con gli stessi indici di ricampionamento: per ciascuna (nome ->
    terms) le righe per unità (frazioni), la media Fraction e il suo float per metrica, i conteggi delle unità con
    valore di ACER < 0, > 0, = 0 e non definito, e l'intervallo bootstrap per soggetto della ricetta del progetto
    (SR.boot_picks, SR.interval; soggetti uniti sui fold; a ogni ricampionamento ogni unità usa le molteplicità dei
    soggetti del suo fold, le stesse per tutte le reti e i due stati; statistica = media dei float delle frazioni per
    unità; ricampionamento scartato per una metrica se il valore di un'unità qualsiasi non è definito)."""
    if not entries:
        fail("comparison: no unit")
    base = [e["ordinary"] for e in entries]
    subjects, index = pooled(base)
    n = len(subjects)
    out = {}
    for name, terms in specs.items():
        rows = [term_value(e, terms) for e in entries]
        mean_f = {k: exact_mean([r[k] for r in rows]) for k in METRICS}
        acer = [r["acer"] for r in rows]
        out[name] = {"rows": [{"unit": e["ordinary"]["unit"], **{k: r[k] for k in METRICS}} for e, r in zip(entries, rows)],
                     "mean_fraction": mean_f, "mean": {k: None if v is None else float(v) for k, v in mean_f.items()},
                     "counts": {"below_zero": sum(1 for x in acer if x is not None and x < 0), "above_zero": sum(1 for x in acer if x is not None and x > 0),
                                "zero": sum(1 for x in acer if x is not None and x == 0), "undefined": sum(1 for x in acer if x is None),
                                "units": len(rows)},
                     "_vals": {k: [] for k in METRICS}}
    boot = {"n_subjects": n, "resamples": SR.BOOT["resamples"], "seed": SR.BOOT["seed"], "percentiles": list(SR.BOOT["percentiles"])}
    if n >= SR.BOOT["min_subjects"]:
        for pick in SR.boot_picks(n):
            w = np.bincount(pick, minlength=n)
            cache = {}
            for name, terms in specs.items():
                per = [term_value(e, terms, w[index[e["ordinary"]["split"]]], cache) for e in entries]
                for k in METRICS:
                    if all(v[k] is not None for v in per):
                        out[name]["_vals"][k].append(float(np.mean([float(v[k]) for v in per])))
    for name in specs:
        vals = out[name].pop("_vals")
        if n < SR.BOOT["min_subjects"]:
            out[name]["interval"] = {k: {"low": None, "high": None, "valid_resamples": 0} for k in METRICS}
            out[name]["note"] = f"no interval: fewer than {SR.BOOT['min_subjects']} subjects"
        else:
            out[name]["interval"] = {k: SR.interval(vals[k]) for k in METRICS}
        out[name]["bootstrap"] = boot
    return out


# ============================================================================ (c) lettura e conclusioni

def contrast_reading(k: str, interval: dict) -> str:
    """Lettura del contrasto C_k dall'intervallo per soggetto del suo ACER sui seed insieme: «… lower than zero» se è
    tutto sotto zero, «… higher than zero» se è tutto sopra, «unresolved» se contiene lo zero (un estremo esattamente a
    zero lo contiene) o non è calcolabile. Estremi non finiti -> Failure (D.bounds). La ragione è D.interval_reason."""
    if k not in ATTACKNETS:
        fail(f"contrast reading: {k!r} is not an AttackNet")
    lo, hi = D.bounds(interval, "contrast reading")
    names = reading_names(k)
    if lo is None or hi is None:
        return names[2]
    if hi < 0:
        return names[0]
    if lo > 0:
        return names[1]
    return names[2]


def conclusion(k: str, reading: str) -> str:
    """La conclusione scritta prima del test per il contrasto di una AttackNet, dalla sua lettura (una per AttackNet,
    nessuna affermazione congiunta)."""
    names = reading_names(k)
    if reading == names[0] or reading == names[1]:
        side = "lower" if reading == names[0] else "higher"
        return (f"On these checkpoints and test frames, the ACER variation with this BN re-estimation (re-estimated minus ordinary BN) is "
                f"algebraically {side} for {k} than for LivenessNet; it does not identify a cause.")
    if reading == names[2]:
        return (f"This comparison does not show that the ACER variation with this BN re-estimation differs between {k} and LivenessNet; "
                "it does not exclude it.")
    fail(f"conclusion: unknown reading {reading!r} for {k}")


def not_computed_text(k: str, pairs: int, reason: str) -> str:
    return f"{NOT_COMPUTED} ({pairs} of {UNITS_REQUIRED} pairs evaluable in both BN states): {reason}"


def not_computed_conclusion(k: str, pairs: int, reason: str) -> str:
    """La conclusione di un contrasto non calcolato (una sola volta il numero delle coppie)."""
    return f"Not computed for {k} ({pairs} of {UNITS_REQUIRED} pairs evaluable in both BN states): {reason}."


# ============================================================================ (k) guardiano per rete

def _combos(units) -> dict:
    """(fold, seed) -> {rete: soglia} delle unità grezze di una condizione."""
    out = {}
    for u in units:
        key = tuple(u["unit"])
        if len(key) != 3:
            fail(f"guardian: unit {key} malformed")
        out[(key[1], key[2])] = {n: v["threshold"] for n, v in u["networks"].items()}
    return out


def guardian(ordinary_units, re_units, reasons=None, threshold_rule: str = PRIMARY_THRESHOLD) -> dict:
    """Il guardiano dell'analisi primaria (e, con threshold_rule «dev_eer», delle condizioni descrittive alla soglia EER)
    per rete, sui valori effettivi, prima di ogni calcolo:
      - con la soglia 0,5, la soglia effettiva di ogni rete presente, in ogni unità e in entrambi gli stati, deve essere
        0,5: altrimenti stop per tutte le reti (Failure); con la soglia EER ogni soglia deve essere un numero finito in
        [0, 1] (D.check_threshold);
      - LivenessNet e ResNet-18 (riusate da C2) devono avere tutte le 12 combinazioni fold × seed in entrambi gli stati:
        altrimenti stop (Failure), come in C2;
      - un'AttackNet con meno di 12 coppie valutabili in entrambi gli stati non ha variazione, contrasto né confronto
        con ResNet-18: «not computed (k of 12 pairs)» con il motivo (reasons: rete -> testo, dato da chi chiama, per
        esempio «not evaluable: no finite val_loss»); le altre reti procedono.
    Nessun contrasto è mai calcolato su meno di 12 coppie."""
    if threshold_rule not in THRESHOLD_RULES:
        fail(f"guardian: unknown threshold rule {threshold_rule!r}")
    want = {(f, s) for f in FOLDS for s in PLAN_SEEDS}
    states = {"ordinary": _combos(ordinary_units), "re-estimated": _combos(re_units)}
    for st, combos in states.items():
        extra = sorted(set(combos) - want)
        if extra:
            fail(f"primary analysis stopped: {st} BN has units outside folds {list(FOLDS)} x seeds {list(PLAN_SEEDS)}")
        for combo, nets in combos.items():
            for n, t in nets.items():
                if n not in NETWORKS:
                    fail(f"primary analysis stopped: unknown network {n!r}")
                t = D.check_threshold(t, f"guardian {n}")
                if threshold_rule == PRIMARY_THRESHOLD and t != float(FR.FIXED_THRESHOLD):
                    fail(f"primary analysis stopped: the effective threshold of {n} is {t}, not 0.5 ({st} BN)")
    out = {}
    for n in NETWORKS:
        have = {st: {c for c, nets in combos.items() if n in nets} for st, combos in states.items()}
        pairs = have["ordinary"] & have["re-estimated"]
        if n in REUSED:
            for st in BN_STATES:
                if have[st] != want:
                    fail(f"primary analysis stopped: {n} (reused from C2) has {len(have[st])} of {UNITS_REQUIRED} fold-seed combinations "
                         f"with {st} BN")
        computed = pairs == want
        reason = None if computed else ((reasons or {}).get(n) or "fewer than 12 fold-seed pairs evaluable in both BN states")
        out[n] = {"pairs": len(pairs), "computed": computed, "reason": reason,
                  "text": "computed" if computed else not_computed_text(n, len(pairs), reason)}
    return out


# ============================================================================ (g) graduatorie, (h) denominatori, (i) punti percentuali

def rankings(pus: list, networks=NETWORKS) -> dict:
    """Graduatoria descrittiva di una condizione: le reti presenti in tutte le unità ordinate per ACER medio (frazioni
    esatte, media sulle unità; pari merito = stesso rango minimo, rango = 1 + numero di reti con ACER medio
    strettamente più basso); per unità i conteggi dei ranghi con la stessa regola sugli ACER dell'unità (un'unità in
    cui l'ACER di una rete non è definito non entra nei conteggi). Nessun intervallo."""
    full = [n for n in networks if all(n in pu["counts"] for pu in pus)]
    means = {}
    for n in full:
        vals = [_rates(pu, n)["acer"] for pu in pus]
        means[n] = exact_mean(vals)
    ranked = [n for n in full if means[n] is not None]

    def rank_of(vals: dict) -> dict:
        return {n: 1 + sum(1 for m in vals if vals[m] < vals[n]) for n in vals}
    r = rank_of({n: means[n] for n in ranked})
    order = sorted(ranked, key=lambda n: (r[n], NETWORKS.index(n)))
    counts = {n: {str(i): 0 for i in range(1, len(ranked) + 1)} for n in ranked}
    skipped = 0
    for pu in pus:
        vals = {n: _rates(pu, n)["acer"] for n in ranked}
        if any(v is None for v in vals.values()):
            skipped += 1
            continue
        for n, k in rank_of(vals).items():
            counts[n][str(k)] += 1
    return {"order": [{"network": n, "rank": r[n], "mean_acer": float(means[n])} for n in order],
            "not_ranked": [n for n in networks if n not in ranked], "unit_rank_counts": counts, "units_not_counted": skipped,
            "units": len(pus)}


def denominators(pus: list) -> dict:
    """I denominatori di una condizione, senza identificativi: unità, soggetti uniti sui fold, frame di test sommati
    sulle unità con gli attacchi (denominatore di APCER) e i bona fide (denominatore di BPCER), e per fold frame,
    attacchi, bona fide e soggetti (uguali per i seed di un fold, prepare)."""
    subjects, _ = pooled(pus)
    per_fold = {}
    tot = {"frames": 0, "attack": 0, "bona_fide": 0}
    for pu in pus:
        c = pu["counts"][pu["networks"][0]]
        na, nb = int(np.sum(c["na"])), int(np.sum(c["nb"]))
        tot["frames"] += pu["n_frames"]
        tot["attack"] += na
        tot["bona_fide"] += nb
        per_fold.setdefault(str(pu["split"][1]), {"frames": pu["n_frames"], "attack": na, "bona_fide": nb, "subjects": len(c["subjects"])})
    return {"units": len(pus), "subjects": len(subjects), "test_frames_summed_over_units": tot["frames"],
            "attack_frames_summed_over_units": tot["attack"], "bona_fide_frames_summed_over_units": tot["bona_fide"], "per_fold": per_fold}


def pp_decimal(x):
    """Una proporzione in punti percentuali: × 100, due decimali, metà lontano dallo zero, sul valore esatto (la regola
    del pacchetto della tabella C2, make_table_figure.points): una Fraction con l'aritmetica esatta, un float dal suo
    valore decimale più breve (repr) con Decimal; None resta None. Solo per la presentazione."""
    if x is None:
        return None
    if not D.finite_real(x):
        fail("percentage points: the value is not a finite number")
    if isinstance(x, Fraction):
        q = abs(x) * 10000
        n = q.numerator // q.denominator
        if (q - n) * 2 >= 1:
            n += 1
        d = Decimal(-n if x < 0 else n).scaleb(-4)
    else:
        d = Decimal(repr(float(x)))
    return (d * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def pp(x):
    """pp_decimal come float (None resta None)."""
    d = pp_decimal(x)
    return None if d is None else float(d)


def pp_text(x) -> str:
    """Testo di una proporzione in punti percentuali con due decimali (vuoto se None)."""
    d = pp_decimal(x)
    return "" if d is None else f"{d:.2f}"


# ============================================================================ (f) classe unica

def one_class_counts(records) -> list:
    """Conteggio di D.one_class_run (dev, soglia 0,5) per rete × patience × stato della BN, per le cinque reti: k modelli
    a una sola classe su n valutabili, i non valutabili a parte. records: {"network", "patience", "bn_state", "result"}."""
    cells = {}
    for r in records:
        if r["network"] not in NETWORKS or r["patience"] not in PATIENCES or r["bn_state"] not in BN_STATES:
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
            for k, v in sorted(cells.items(), key=lambda kv: (NETWORKS.index(kv[0][0]), kv[0][1], BN_STATES.index(kv[0][2])))]


def test_one_class(y, s, threshold: float) -> dict:
    """Dato descrittivo sul test: un modello che, alla soglia data, decide tutti i frame attacco o tutti bona fide (con
    la regola del progetto, D.attack_decisions). Non entra in nessuna lettura."""
    y, s = D.labels_scores(y, s, "test")
    t = D.check_threshold(threshold, "test one-class")
    k = int(np.sum(D.attack_decisions(s, t)))
    cls = None if not len(y) else ("bona fide" if k == 0 else "attack" if k == len(y) else None)
    return {"n": int(len(y)), "n_decided_attack": k, "one_class": cls is not None, "class": cls}


# ============================================================================ analisi primaria e descrittive

def _public(q: dict) -> dict:
    return {"mean": q["mean"], "interval": q["interval"], "counts": q["counts"]}


def c2_gap_terms(state: str) -> tuple:
    """Divario di C2 nel suo verso, metrica(ResNet-18) − metrica(LivenessNet), nello stato dato."""
    return ((1, EXTERNAL, state), (-1, REFERENCE, state))


def c2_contrast_terms() -> tuple:
    """Contrasto di C2 nel suo verso: (R_re − L_re) − (R_ord − L_ord), cioè V_R − V_L."""
    return ((1, EXTERNAL, "re-estimated"), (-1, EXTERNAL, "ordinary"), (-1, REFERENCE, "re-estimated"), (1, REFERENCE, "ordinary"))


def negated(q: dict) -> dict:
    """Una quantità calcolata nel verso di C2 riportata nel verso opposto: media e righe negate (esatte), estremi negati e
    scambiati (low = −high, high = −low: negare i float è esatto), conteggi sotto e sopra lo zero scambiati. Così
    LivenessNet contro ResNet-18 è esattamente l'opposto dei valori di C2, intervalli compresi."""
    neg = lambda v: None if v is None else -v          # noqa: E731
    out = {"rows": [{"unit": r["unit"], **{k: neg(r[k]) for k in METRICS}} for r in q["rows"]],
           "mean_fraction": {k: neg(v) for k, v in q["mean_fraction"].items()}, "mean": {k: neg(v) for k, v in q["mean"].items()},
           "interval": {k: {"low": neg(v["high"]), "high": neg(v["low"]), "valid_resamples": v["valid_resamples"]} for k, v in q["interval"].items()},
           "counts": {**q["counts"], "below_zero": q["counts"]["above_zero"], "above_zero": q["counts"]["below_zero"]},
           "bootstrap": q["bootstrap"]}
    if "note" in q:
        out["note"] = q["note"]
    return out


def analyse(entries: list, status: dict, readings: bool = True) -> dict:
    """Variazioni delle cinque reti, contrasti delle tre AttackNet con LivenessNet (con readings, lettura e conclusione;
    senza, solo i valori: condizioni descrittive) e confronto descrittivo con ResNet-18 sulle unità appaiate date (tutte
    o quelle di un seed); status: l'uscita di guardian. Reti «not computed» escluse da ogni quantità; stessi indici di
    ricampionamento per tutto. Per LivenessNet il confronto con ResNet-18 si calcola nel verso di C2 (ResNet-18 meno
    LivenessNet) e si riporta negato (negated): è l'opposto esatto dei divari e del contrasto di C2."""
    specs = {}
    for n in NETWORKS:
        if status[n]["computed"]:
            specs[("variation", n)] = variation_terms(n)
    for k in ATTACKNETS:
        if status[k]["computed"]:
            specs[("contrast", k)] = contrast_terms(k)
    for n in ATTACKNETS:
        if status[n]["computed"]:
            for st in BN_STATES:
                specs[("gap", n, st)] = gap_terms(n, st)
            specs[("variation_difference", n)] = variation_difference_terms(n)
    for st in BN_STATES:
        specs[("c2_gap", st)] = c2_gap_terms(st)
    specs[("c2_contrast",)] = c2_contrast_terms()
    q = quantities(entries, specs)
    for st in BN_STATES:
        q[("gap", REFERENCE, st)] = negated(q[("c2_gap", st)])
    q[("variation_difference", REFERENCE)] = negated(q[("c2_contrast",)])
    out = {"variations": {}, "contrasts": {}, "descriptive_vs_resnet18": {}}
    for n in NETWORKS:
        out["variations"][n] = {**_public(q[("variation", n)]), "computed": True} if status[n]["computed"] else \
            {"computed": False, "text": status[n]["text"]}
    for k in ATTACKNETS:
        if not status[k]["computed"]:
            out["contrasts"][k] = {"computed": False, "text": status[k]["text"], "reading": NOT_COMPUTED,
                                   "conclusion": not_computed_conclusion(k, status[k]["pairs"], status[k]["reason"])}
            continue
        c = q[("contrast", k)]
        out["contrasts"][k] = {**_public(c), "computed": True}
        if readings:
            rd = contrast_reading(k, c["interval"]["acer"])
            out["contrasts"][k].update({"reading": rd, "reason": D.interval_reason(c["interval"]["acer"]), "conclusion": conclusion(k, rd)})
    for n in OURS:
        if not status[n]["computed"]:
            out["descriptive_vs_resnet18"][n] = {"computed": False, "text": status[n]["text"]}
            continue
        out["descriptive_vs_resnet18"][n] = {"computed": True, **{f"gap_{st.replace('-', '_')}": _public(q[("gap", n, st)]) for st in BN_STATES},
                                             "variation_difference": _public(q[("variation_difference", n)])}
    out["n_subjects"] = q[("c2_contrast",)]["bootstrap"]["n_subjects"]
    out["_quantities"] = {k: v for k, v in q.items() if k[0] not in ("c2_gap", "c2_contrast")}
    return out


def _check_conditions(conditions) -> None:
    if not isinstance(conditions, dict) or any(k not in CONDITIONS for k in conditions):
        fail(f"analysis: conditions must be keyed by (patience, BN state, threshold) among the {len(CONDITIONS)} conditions")


def pair_analysis(conditions: dict, patience: int, threshold_rule: str, reasons=None, readings: bool = False) -> dict:
    """Le stesse quantità della primaria su una coppia di condizioni (patience, soglia) nei due stati della BN: guardiano
    (con la regola della soglia data), unità in ordine (fold, seed), analyse su tutte le unità e per seed; denominatori."""
    o, r = (patience, "ordinary", threshold_rule), (patience, "re-estimated", threshold_rule)
    for k in (o, r):
        if k not in conditions:
            fail(f"analysis: the condition {k} is missing")
    status = guardian(conditions[o], conditions[r], reasons, threshold_rule)
    order = lambda units: sorted(units, key=lambda u: (u["unit"][1], u["unit"][2]))       # noqa: E731
    ordinary = prepare(order(conditions[o]), REUSED)
    re_est = prepare(order(conditions[r]), REUSED)
    entries = pair_states(ordinary, re_est)
    for e in entries:
        for k in ATTACKNETS:
            if status[k]["computed"] and any(k not in e[s]["counts"] for s in BN_STATES):
                fail(f"analysis: {k} is missing from unit {e['ordinary']['unit']}")
    res = analyse(entries, status, readings)
    q = res.pop("_quantities")
    per_seed = {}
    for seed in PLAN_SEEDS:
        sub = [e for e in entries if e["ordinary"]["seed"] == seed]
        rr = analyse(sub, status, readings)
        rr.pop("_quantities")
        per_seed[str(seed)] = rr
    rows = []
    for i, e in enumerate(entries):
        rows.append({"unit": e["ordinary"]["unit"], "n_frames": e["ordinary"]["n_frames"], "frame_ids_sha256": e["ordinary"]["frame_ids_sha256"],
                     **{"/".join(name): {m: q[name]["rows"][i][m] for m in METRICS} for name in q}})
    return {"conditions": {"ordinary": list(o), "re-estimated": list(r)}, "intervals": INTERVALS_NOTE, "presentation": PP_NOTE,
            "guardian": status, **res, "per_seed": per_seed,
            "denominators": {st: denominators(p) for st, p in (("ordinary", ordinary), ("re-estimated", re_est))},
            "rows": rows, "_prepared": {"ordinary": ordinary, "re-estimated": re_est}}


def primary_analysis(conditions: dict, reasons=None) -> dict:
    """L'analisi primaria fissata dal coordinatore sulla scheda C5 il 21/11/2026: «Con patience 4 e soglia 0,5, calcoli
    per ciascuna rete la variazione di ACER (BN ristimata meno ordinaria, in punti percentuali). Per ogni AttackNet
    confronti questa variazione con quella di LivenessNet, sulle stesse 12 coppie fold–seed e sugli stessi soggetti.
    Riporti tutti e tre i contrasti con intervalli bootstrap appaiati per soggetto, specificando che gli IC al 95% sono
    individuali.»
    conditions: {(patience, stato BN, soglia): unità} con chiavi fra le 8 CONDITIONS; la funzione sceglie da sé
    PRIMARY_ORDINARY e PRIMARY_RE_ESTIMATED (una delle due assente -> Failure). Prima di ogni calcolo il guardiano
    (guardian); poi le unità dei due stati (prepare, LivenessNet e ResNet-18 obbligatorie) in ordine (fold, seed) e
    appaiate (pair_states); analyse con le letture su tutte le unità e, a parte, per ciascun seed (fuori dalla lettura);
    denominatori e graduatorie descrittive dei due stati. I valori sono proporzioni a precisione piena: i punti
    percentuali si ricavano in presentazione (pp)."""
    _check_conditions(conditions)
    res = pair_analysis(conditions, PRIMARY_PATIENCE, PRIMARY_THRESHOLD, reasons, readings=True)
    prep = res.pop("_prepared")
    return {**res, "definition": ("variation of a network = metric(re-estimated BN) - metric(ordinary BN); contrast of an AttackNet = its "
                                  "variation minus the variation of LivenessNet, per unit, as exact fractions"),
            "rankings": {st: rankings(p) for st, p in prep.items()}}


def descriptive_analysis(conditions: dict, reasons=None) -> dict:
    """(o) Le condizioni descrittive: le stesse quantità di (a) e (m), con i loro intervalli e senza alcuna lettura, a
    patience 12 con soglia 0,5 e alla soglia EER a entrambe le patience (pair_analysis, readings False; il guardiano
    alla soglia EER non chiede 0,5); le graduatorie (g) per ciascuna delle 8 condizioni. Tutte le 8 condizioni sono
    obbligatorie: una condizione assente ferma la valutazione (Failure)."""
    _check_conditions(conditions)
    missing = [c for c in CONDITIONS if c not in conditions]
    if missing:
        fail(f"descriptive analysis: the conditions {missing} are missing (all 8 are required)")
    pairs = {}
    for p, t in ((12, PRIMARY_THRESHOLD), (4, "dev_eer"), (12, "dev_eer")):
        r = pair_analysis(conditions, p, t, reasons, readings=False)
        r.pop("_prepared")
        pairs[f"patience_{p}_{t}"] = r
    ranks = {}
    for c in CONDITIONS:
        units = sorted(conditions[c], key=lambda u: (u["unit"][1], u["unit"][2]))
        ranks["/".join(map(str, c))] = rankings(prepare(units, REUSED))
    return {"pairs": pairs, "rankings": ranks, "intervals": INTERVALS_NOTE, "note": "descriptive: no reading"}
