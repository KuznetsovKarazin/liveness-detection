"""
Definizioni operative del confronto tra reti PAD (LivenessNet e ResNet-18; incarico del coordinatore sulla scheda C2,
17/11/2026; analisi primaria e controllo degli ID dei frame chiesti dal coordinatore il 19/11/2026), come funzioni
pure. In questa consegna non sono eseguite su dati reali: fissano le definizioni e sono provate su dati sintetici
(scripts/check_pad_arch_comparison.py).

  (a) one_class_run, one_class_counts   run che a soglia 0,5 assegnano tutto il dev a una classe
  (b) compare_condition                 differenza appaiata ResNet-18 − LivenessNet di ACER, APCER e BPCER per unità,
                                        pareggi esatti, media, conteggi, intervallo bootstrap per soggetto, per seed,
                                        con la lettura per seed (per_seed_reading)
  (c) sign_of, compare_conditions       segno di una condizione e cambio di segno fra due condizioni, con l'inversione
                                        della sola stima puntuale, il contrasto appaiato e le unità che cambiano segno
  (d) leave_one_subject_out             sensibilità del segno ai soggetti, scelta dell'autore nella forma del protocollo
                                        N1 del coordinatore (24/10/2026):
                                        media di ΔACER ricalcolata escludendo un soggetto di test alla volta; riportata
                                        accanto al segno, non entra nelle regole di (c)
  (e) primary_analysis                  l'analisi primaria fissata dal coordinatore il 19/11/2026: patience 4, soglia
                                        0,5; Δ con BN ordinaria e con BN ristimata sul solo fitting e il contrasto
                                        Δ(ristimata) − Δ(ordinaria), su tutte le unità e, a parte, per seed; le altre
                                        sei condizioni (8 in tutto, 2 nell'analisi primaria) e ogni confronto fra
                                        condizioni restano analisi secondarie, descrittive
  (f) per_seed_reading                  lettura per seed (la regola della PR #21, nuaa_session_ablation_run.
                                        reading_verdict) sui seed previsti PLAN_SEEDS (42, 43, 44): stesso segno non
                                        nullo per ogni seed e ogni intervallo per seed che esclude lo zero da quel
                                        lato; lettura secondaria, con il suo nome

ID dei frame (richiesta del coordinatore del 19/11/2026): ogni rete di ogni unità porta, accanto a etichette, punteggi
e soggetti, il campo FRAME_ID_FIELD («sample_id», il nome della chiave del frame nei file dei punteggi dei run
accettati, nuaa_folds_run.SCORE_COLUMNS e nuaa_session_ablation_run.score_rows) con l'ID di ciascun frame, uno per
punteggio. Tutti i controlli degli ID passano da prepare_units (un solo punto del codice, usato da compare_condition,
leave_one_subject_out e quindi da compare_conditions e primary_analysis) e da same_frames (fra due condizioni).
Nessun messaggio d'errore e nessuna uscita riepilogativa contiene ID di frame: nelle righe per unità ci sono solo il
numero dei frame e l'impronta SHA-256 della lista ordinata degli ID (frame_ids_sha256). Il campo interno «prepared»
(restituito da compare_condition per compare_conditions) contiene ID di frame e di soggetto e non è un'uscita
pubblicabile; compare_conditions ne controlla solo la forma (check_prepared) e non lo rivalida: chi lo manomette dopo
compare_condition aggira i controlli. Anche leave_one_subject_out riporta il soggetto escluso (excluded_subject), come
nella consegna del 18/11.

Regola di decisione del progetto in un solo punto (attack_decisions): attacco se score > soglia, pareggio = bona fide
(nuaa_folds_run.DECISION_RULE). Soglia EER: nuaa_folds_run.eer_threshold sul dev, invariata (condition_threshold).
Bootstrap: la ricetta del progetto (nuaa_session_ablation_run.BOOT, boot_picks, interval: 1.000 ricampionamenti,
default_rng(0), percentili 2,5-97,5, minimo 5 soggetti), con la statistica e la regola di scarto di nuaa_bn_test
(media sulle unità delle differenze appaiate, stessi indici per le due reti; ricampionamento scartato se la metrica di
un'unità qualsiasi non è definita; ricampionamenti validi riportati).
Le letture non nominano cause. Valori fuori dominio (etichette diverse da 0/1, punteggi non finiti o fuori da [0, 1],
soglia fuori da [0, 1], campi di una rete diversi da quelli fissati, ID dei frame mancanti, vuoti, duplicati o di
numero diverso dai punteggi, frame diversi fra le due reti, fra unità dello stesso split o fra due condizioni, un ID
in due split, unità duplicate, condizioni con unità diverse) -> Failure.
"""
import hashlib
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
FRAME_ID_FIELD = "sample_id"                              # la chiave del frame nei file dei punteggi dei run accettati
NETWORK_FIELDS = ("y", "s", "subject", "threshold", FRAME_ID_FIELD)
# analisi primaria (coordinatore, 19/11/2026): patience 4, soglia 0,5; contrasto = Δ(BN ristimata) − Δ(BN ordinaria)
PRIMARY_PATIENCE, PRIMARY_THRESHOLD = 4, "fixed_0.5"
PRIMARY_ORDINARY = (PRIMARY_PATIENCE, "ordinary", PRIMARY_THRESHOLD)          # sottraendo del contrasto
PRIMARY_RE_ESTIMATED = (PRIMARY_PATIENCE, "re-estimated", PRIMARY_THRESHOLD)  # minuendo del contrasto
CONTRAST_LOWER = "ACER difference (ResNet-18 - LivenessNet) lower with re-estimated BN"
CONTRAST_HIGHER = "ACER difference (ResNet-18 - LivenessNet) higher with re-estimated BN"
CONTRAST_UNRESOLVED = "unresolved"
VERDICTS = SR.VERDICTS                                    # ("consistent", "inconclusive"), la regola della PR #21
PLAN_SEEDS = tuple(SR.TRAIN_SEEDS)                        # i seed previsti dal piano (42, 43, 44): un solo punto
INTERVAL_REASONS = ("interval entirely below zero", "interval entirely above zero", "interval contains zero",
                    "no interval (fewer subjects than the minimum, or no valid resample)")


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


def frame_ids(v, n: int, what: str) -> tuple:
    """Gli ID dei frame di una rete (FRAME_ID_FIELD): lista, tupla o array 1-D di n testi non vuoti, tutti diversi
    (uno per punteggio, nell'ordine dei punteggi); altrimenti Failure, senza alcun ID nel messaggio."""
    if not isinstance(v, (list, tuple, np.ndarray)) or (isinstance(v, np.ndarray) and v.ndim != 1):
        fail(f"{what}: {FRAME_ID_FIELD} must be a list with one frame identifier per score")
    ids = tuple(v.tolist() if isinstance(v, np.ndarray) else v)
    if len(ids) != n:
        fail(f"{what}: {len(ids)} frame identifiers for {n} scores (one per score is required)")
    if any(not isinstance(x, str) or not x.strip() for x in ids):
        fail(f"{what}: every frame identifier must be a non-empty text")
    if len(set(ids)) != len(ids):
        fail(f"{what}: duplicated frame identifiers")
    return ids


def ids_sha256(ids) -> str:
    """Impronta pubblicabile di una lista di ID dei frame: SHA-256 degli ID ordinati uniti da a capo, in UTF-8 (la forma
    di nuaa_bn_recalibration.keys_sha)."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def prepare_units(units) -> list:
    """Unità di una condizione: {"unit": (dataset, fold o split, seed), "networks": {rete: {"y", "s", "subject",
    "threshold", "sample_id"}}} (esattamente questi campi per ciascuna delle due reti). Unico punto dei controlli sui
    frame di una condizione:
      - le due reti di un'unità hanno le stesse etichette e gli stessi soggetti nello stesso ordine e, controllo a
        parte, gli stessi ID dei frame nello stesso ordine (frame_ids: uno per punteggio, non vuoti, senza duplicati);
      - unità tutte diverse;
      - due unità dello stesso split (seed diversi) sono valutate sugli stessi frame: la stessa sequenza di (ID,
        soggetto, etichetta), cioè stessi ID nello stesso ordine e con lo stesso soggetto e la stessa etichetta per
        ciascun ID (la regola più stretta: nei run accettati i punteggi del test di un fold sono scritti nell'ordine
        dei frame assemblati, cioè delle righe del fold ordinate per chiave, nuaa_folds_run.assemble e
        nuaa_session_ablation_run.score_rows, un ordine che non dipende dal seed);
      - un ID dei frame compare in un solo split (i test dei fold sono disgiunti);
      - il seed di ogni unità è un intero (int di Python non booleano o intero di NumPy) fra i seed del piano
        (PLAN_SEEDS): 42.0 è rifiutato;
      - tutte le unità di una chiamata sono dello stesso dataset: una condizione si confronta per dataset (le regole
        sugli split, sugli ID dei frame e sull'unione dei soggetti valgono sulle unità della chiamata).
    Restituisce per unità i conteggi per soggetto, la sequenza dei frame (campo interno «frames»: ID, soggetti,
    etichette; non pubblicabile), il numero dei frame e l'impronta degli ID (ids_sha256)."""
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
            fail("comparison: the units of one call must belong to one dataset (a condition is compared per dataset)")
        seen.add(key)
        nets = u["networks"]
        if sorted(nets) != sorted(AM.NETWORKS):
            fail(f"comparison: unit {key} must hold exactly {', '.join(AM.NETWORKS)}")
        for n in AM.NETWORKS:
            if not isinstance(nets[n], dict) or sorted(nets[n]) != sorted(NETWORK_FIELDS):
                fail(f"comparison: unit {key} {n} must hold exactly the fields {', '.join(NETWORK_FIELDS)}")
        a, b = nets[MINUEND], nets[SUBTRAHEND]
        if len(a["y"]) != len(b["y"]) or not np.array_equal(np.asarray(a["y"]), np.asarray(b["y"])) \
                or not np.array_equal(np.asarray(a["subject"]).astype(str), np.asarray(b["subject"]).astype(str)):
            fail(f"comparison: unit {key}: the two networks are not scored on the same test frames")
        counts = {n: subject_counts(nets[n]["y"], nets[n]["s"], nets[n]["subject"], check_threshold(nets[n]["threshold"], f"unit {key} {n}"))
                  for n in AM.NETWORKS}
        if not counts[MINUEND]["subjects"]:
            fail(f"comparison: unit {key} has no test frame")
        ids = {n: frame_ids(nets[n][FRAME_ID_FIELD], len(nets[n]["y"]), f"comparison: unit {key} {n}") for n in AM.NETWORKS}
        if ids[MINUEND] != ids[SUBTRAHEND]:
            fail(f"comparison: unit {key}: the two networks are not scored on the same frame identifiers in the same order")
        frames = (ids[MINUEND], tuple(np.asarray(a["subject"]).astype(str).tolist()), tuple(int(x) for x in np.asarray(a["y"])))
        split = (key[0], key[1])
        if split not in split_frames:
            for i in frames[0]:
                if id_split.setdefault(i, split) != split:
                    fail(f"comparison: a frame identifier of {split} also appears in {id_split[i]}")
        if split_frames.setdefault(split, frames) != frames:
            fail(f"comparison: two units of {split} are not scored on the same frames (identifiers, subjects and labels in the same order)")
        out.append({"unit": key, "split": split, "seed": key[2], "counts": counts, "frames": frames, "n_frames": len(frames[0]),
                    "frame_ids_sha256": ids_sha256(frames[0])})
    return out


def same_frames(x: dict, y: dict, what: str) -> None:
    """Due unità preparate (prepare_units) della stessa unità in due condizioni: stessi soggetti e stesse etichette
    nello stesso ordine e, controllo a parte, stessi ID dei frame nello stesso ordine; altrimenti Failure (nessun ID nel
    messaggio)."""
    if x["frames"][1:] != y["frames"][1:]:
        fail(f"{what}: unit {x['unit']} is not scored on the same test frames (subjects and labels) in the two conditions")
    if x["frames"][0] != y["frames"][0]:
        fail(f"{what}: unit {x['unit']}: the two conditions are not scored on the same frame identifiers in the same order")


def unit_row(pu: dict) -> dict:
    """Riga di un'unità: metriche esatte delle due reti, Δ (ResNet-18 − LivenessNet) e pareggio di ACER come frazioni;
    numero dei frame e impronta dei loro ID."""
    ones = {n: np.ones(len(pu["counts"][n]["subjects"]), np.int64) for n in AM.NETWORKS}
    m = {n: weighted_rates(pu["counts"][n], ones[n]) for n in AM.NETWORKS}
    d = {k: diff(m[MINUEND][k], m[SUBTRAHEND][k]) for k in METRICS}
    return {"unit": pu["unit"], "metrics": m, "delta": d, "tie": None if d["acer"] is None else d["acer"] == 0,
            "n_frames": pu["n_frames"], "frame_ids_sha256": pu["frame_ids_sha256"]}


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
    validi riportati). Il segno della condizione è sign_of(intervallo di ACER), con accanto sign_reason
    (interval_reason: distingue l'intervallo che contiene lo zero da quello non calcolabile; campo in più, i nomi del
    segno sono invariati). Accanto, con il suo nome, la lettura per seed (per_seed_reading) di ΔACER sui valori di
    ciascuno dei seed del piano PLAN_SEEDS (media e intervallo di quel seed; None per un seed senza unità, quindi mai
    «consistent» con un seed mancante), lettura secondaria che non entra in sign_of. Il campo «prepared» (le unità di
    prepare_units, con ID di frame e soggetti) serve a compare_conditions e non è un'uscita pubblicabile."""
    pus = prepare_units(units)
    out = condition_of(pus)
    out["per_seed"] = {}
    for seed in sorted({pu["seed"] for pu in pus}):
        sub = summarise([pu for pu in pus if pu["seed"] == seed])
        sub["sign"] = sign_of(sub["interval"]["acer"])
        sub["sign_reason"] = interval_reason(sub["interval"]["acer"])
        out["per_seed"][seed] = sub
    out["per_seed_reading"] = per_seed_reading([{"difference": out["per_seed"][s]["mean_delta"]["acer"],
                                                 "low": out["per_seed"][s]["interval"]["acer"]["low"],
                                                 "high": out["per_seed"][s]["interval"]["acer"]["high"]} if s in out["per_seed"] else None
                                                for s in PLAN_SEEDS])
    return out


def condition_of(pus: list) -> dict:
    """Riepilogo (summarise) di unità già preparate da prepare_units, con il segno sign_of, la sua ragione
    (interval_reason) e le unità stesse (campo interno «prepared», non rivalidato): la forma che compare_conditions
    riceve."""
    out = summarise(pus)
    out["sign"] = sign_of(out["interval"]["acer"])
    out["sign_reason"] = interval_reason(out["interval"]["acer"])
    out["prepared"] = pus
    return out


# ============================================================================ (c) segno e cambio di segno

def finite_real(v) -> bool:
    """Un reale finito: int, float o Fraction di Python, o intero o reale di NumPy, non booleano."""
    return isinstance(v, (int, float, Fraction, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) and math.isfinite(v)


def bounds(interval: dict, what: str) -> tuple:
    """Gli estremi (low, high) di un intervallo: entrambi None (intervallo non calcolabile) oppure due reali finiti con
    low <= high; altrimenti Failure (un NaN darebbe una lettura falsa)."""
    lo, hi = interval.get("low"), interval.get("high")
    if lo is None and hi is None:
        return None, None
    if not finite_real(lo) or not finite_real(hi) or lo > hi:
        fail(f"{what}: the interval bounds must be two finite numbers with low <= high, or both None")
    return lo, hi


def sign_of(interval: dict) -> str:
    """(c) Segno di una condizione dall'intervallo di ΔACER: «ResNet-18 lower» se è tutto sotto zero, «LivenessNet
    lower» se è tutto sopra, «unresolved» se contiene lo zero (un estremo esattamente a zero lo contiene) o non è
    calcolabile. Estremi non finiti -> Failure (bounds)."""
    lo, hi = bounds(interval, "sign")
    if lo is None or hi is None:
        return SIGN_UNRESOLVED
    if hi < 0:
        return SIGN_RESNET_LOWER
    if lo > 0:
        return SIGN_LIVENESSNET_LOWER
    return SIGN_UNRESOLVED


def interval_reason(interval: dict) -> str:
    """La ragione di una lettura fatta da un intervallo (sign_of, contrast_reading): uno di INTERVAL_REASONS, «interval
    entirely below zero», «interval entirely above zero», «interval contains zero» (un estremo esattamente a zero lo
    contiene), «no interval (…)» (meno soggetti del minimo, oppure nessun ricampionamento valido perché la metrica non
    è definita). Testi senza cause. Estremi non finiti -> Failure (bounds)."""
    lo, hi = bounds(interval, "interval reason")
    if lo is None or hi is None:
        return INTERVAL_REASONS[3]
    if hi < 0:
        return INTERVAL_REASONS[0]
    if lo > 0:
        return INTERVAL_REASONS[1]
    return INTERVAL_REASONS[2]


def per_seed_reading(per_seed: list) -> dict:
    """(f) Lettura per seed, secondaria e con il suo nome (protocollo, §6): la regola della PR #21
    (nuaa_session_ablation_run.reading_verdict, nella stessa forma: verdetti «consistent» / «inconclusive», con le sue
    ragioni, salvo una: un seed senza valore dà «a seed has no value» invece di «a seed has no pooled value (incomplete
    run)», perché qui il valore di un seed è la media sulle sue unità e non un valore unito sui fold) applicata a ΔACER =
    ACER(ResNet-18) − ACER(LivenessNet). per_seed: esattamente un elemento per ciascuno dei seed del piano PLAN_SEEDS,
    nel loro ordine (altrimenti Failure): {"difference": media di ΔACER sulle unità del seed, "low", "high": il suo
    intervallo bootstrap per soggetto} oppure None per un seed senza unità. «consistent» solo se la differenza ha lo stesso segno non nullo in ogni seed E l'intervallo di ogni seed
    esclude lo zero dal lato di quel segno (un estremo a zero non lo esclude); la direzione è «ResNet-18 lower» (segno
    negativo) o «LivenessNet lower» (positivo); altrimenti «inconclusive», anche quando a un seed manca il valore, la
    differenza o l'intervallo. Una differenza o un estremo non finito (NaN, infinito) -> Failure. Non entra in sign_of
    né nel cambio di segno."""
    if not isinstance(per_seed, list) or len(per_seed) != len(PLAN_SEEDS):
        fail(f"per-seed reading: one value (or None) per seed of the plan {list(PLAN_SEEDS)} is required")
    for e in per_seed:
        if e is not None and ((e["difference"] is not None and not finite_real(e["difference"]))
                              or any(e[k] is not None and not finite_real(e[k]) for k in ("low", "high"))):
            fail("per-seed reading: a difference or an interval bound is not a finite number")
    out = {"verdict": "inconclusive", "direction": None}
    if any(e is None for e in per_seed):
        return {**out, "reason": "a seed has no value"}
    d = [e["difference"] for e in per_seed]
    if any(x is None for x in d):
        return {**out, "reason": "a seed has no difference (a class absent)"}
    if any(e["low"] is None or e["high"] is None for e in per_seed):
        return {**out, "reason": "a seed has no bootstrap interval"}
    signs = {sign_value(x) for x in d}
    if 0 in signs:
        return {**out, "reason": "a seed has a zero difference"}
    if len(signs) != 1:
        return {**out, "reason": "the sign differs across the seeds"}
    sign = signs.pop()
    if not all((e["low"] > 0) if sign > 0 else (e["high"] < 0) for e in per_seed):
        return {**out, "reason": "the bootstrap interval of a seed does not exclude 0 on the side of the difference"}
    return {"verdict": VERDICTS[0], "direction": SIGN_LIVENESSNET_LOWER if sign > 0 else SIGN_RESNET_LOWER,
            "reason": "same sign for every seed and every interval excludes 0"}


def check_prepared(c, what: str) -> list:
    """La forma del campo interno «prepared» di una condizione (lista non vuota di unità con unit, split, seed, counts,
    frames, n_frames, frame_ids_sha256; frames con la forma di frames_shape); altrimenti Failure. Solo la forma: il
    contenuto non è rivalidato."""
    keys = {"unit", "split", "seed", "counts", "frames", "n_frames", "frame_ids_sha256"}
    pus = c.get("prepared") if isinstance(c, dict) else None
    if not isinstance(pus, list) or not pus or any(not isinstance(p, dict) or not keys <= set(p) for p in pus) \
            or not all(k in c for k in ("sign", "mean_delta")) or any(not frames_shape(p["frames"]) for p in pus):
        fail(f"sign change: condition {what} is not the output of compare_condition (field prepared malformed)")
    return pus


def frames_shape(f) -> bool:
    """La forma del campo interno «frames»: tre tuple (ID, soggetti, etichette) della stessa lunghezza non nulla."""
    return isinstance(f, tuple) and len(f) == 3 and all(isinstance(x, tuple) for x in f) and len(f[0]) > 0 \
        and len(f[0]) == len(f[1]) == len(f[2])


def compare_conditions(a: dict, b: dict) -> dict:
    """(c) Due condizioni già confrontate con compare_condition (o condition_of), sulle stesse unità nello stesso ordine
    e, per ogni unità, sugli stessi frame (same_frames: stessi soggetti ed etichette e, a parte, stessi ID dei frame,
    nello stesso ordine). Cambio di segno riconosciuto solo se entrambe sono risolte e di segno opposto. A parte, con
    nomi diversi: l'inversione della sola stima puntuale (medie di ΔACER di segno opposto, entrambe diverse da zero, con
    almeno una condizione non risolta); il contrasto appaiato Δ_a − Δ_b (media sulle unità e intervallo con gli stessi
    indici di ricampionamento per le due condizioni e le due reti); il numero di unità il cui ΔACER ha segno
    strettamente opposto nelle due condizioni. Nessuna lettura nomina cause. Di «prepared» è controllata solo la forma
    (check_prepared, altrimenti Failure)."""
    pa, pb = check_prepared(a, "a"), check_prepared(b, "b")
    if [p["unit"] for p in pa] != [p["unit"] for p in pb]:
        fail("sign change: the two conditions are not on the same units")
    for x, y in zip(pa, pb):
        same_frames(x, y, "sign change")
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


# ============================================================================ (e) analisi primaria

def contrast_reading(interval: dict) -> str:
    """Lettura del contrasto primario (scelta dell'autore, non del coordinatore): dall'intervallo bootstrap per soggetto
    del contrasto di ACER sui seed insieme, CONTRAST_LOWER se è tutto sotto zero, CONTRAST_HIGHER se è tutto sopra,
    CONTRAST_UNRESOLVED se contiene lo zero (un estremo esattamente a zero lo contiene) o non è calcolabile. Un contrasto
    negativo vuol dire che la differenza con segno Δ = ACER(ResNet-18) − ACER(LivenessNet) è algebricamente più bassa
    con la BN ristimata che con la BN ordinaria; non dice quale rete ha l'ACER più basso in ciascuno stato della BN e
    non nomina cause. La ragione (interval_reason) è riportata accanto, in contrast_summary: distingue l'intervallo che
    contiene lo zero da quello non calcolabile. Estremi non finiti -> Failure (bounds)."""
    lo, hi = bounds(interval, "contrast reading")
    if lo is None or hi is None:
        return CONTRAST_UNRESOLVED
    if hi < 0:
        return CONTRAST_LOWER
    if lo > 0:
        return CONTRAST_HIGHER
    return CONTRAST_UNRESOLVED


def gap_summary(c: dict) -> dict:
    """La parte pubblicabile del riepilogo di una condizione (nessun ID): media di Δ, intervallo, conteggi di ΔACER, segno
    con la sua ragione, soggetti del bootstrap."""
    return {"mean_delta": c["mean_delta"], "interval": c["interval"], "acer_counts": c["acer_counts"], "sign": c["sign"],
            "sign_reason": c["sign_reason"], "n_subjects": c["bootstrap"]["n_subjects"]}


def contrast_summary(c: dict) -> dict:
    """La parte pubblicabile di compare_conditions(ristimata, ordinaria) con la lettura del contrasto di ACER e la sua
    ragione (interval_reason dello stesso intervallo di ACER)."""
    return {"mean": c["contrast_mean"], "interval": c["contrast_interval"], "units": c["units"],
            "units_changing_sign": c["units_changing_sign"], "reading": contrast_reading(c["contrast_interval"]["acer"]),
            "reason": interval_reason(c["contrast_interval"]["acer"])}


def primary_analysis(conditions: dict) -> dict:
    """(e) L'analisi primaria fissata dal coordinatore il 19/11/2026: «patience 4 e soglia 0,5. Calcoli il divario di ACER
    (ResNet-18 meno LivenessNet) con BN ordinaria e con BN ristimata sui soli dati di fitting, poi sottragga il primo
    divario dal secondo. Riporti a parte i risultati per seed.»
    conditions: {(patience, stato BN, soglia): unità} con chiavi fra le 8 CONDITIONS (le altre sono ammesse e ignorate);
    la funzione sceglie da sé le due condizioni primarie, PRIMARY_ORDINARY = (4, "ordinary", "fixed_0.5") e
    PRIMARY_RE_ESTIMATED = (4, "re-estimated", "fixed_0.5"): l'ordine della sottrazione non dipende da chi chiama; una
    delle due assente, una chiave fuori da CONDITIONS -> Failure.
    Per ciascuna delle due: compare_condition (Δ = metrica(ResNet-18) − metrica(LivenessNet) per ACER, APCER e BPCER;
    media sulle unità, intervallo bootstrap per soggetto, conteggi, segno). Contrasto primario = Δ(BN ristimata) −
    Δ(BN ordinaria), per metrica: compare_conditions(ristimata, ordinaria), quindi stesse unità e stessi frame (anche gli
    ID) nelle due condizioni, media sulle unità e intervallo con gli stessi indici di ricampionamento per le due
    condizioni e le due reti. Righe per unità: Δ nelle due condizioni e contrasto, come frazioni esatte, con il numero
    dei frame e l'impronta degli ID. Lettura del contrasto (contrast_reading, scelta dell'autore) solo dall'intervallo
    del contrasto di ACER sui seed insieme. A parte, per ciascun seed: gli stessi valori (divari, contrasto, segni e
    lettura calcolati sulle sole unità del seed); i valori per seed non entrano nella lettura. Le altre sei condizioni
    (8 in tutto) non sono usate qui: restano analisi secondarie, descrittive."""
    if not isinstance(conditions, dict) or any(k not in CONDITIONS for k in conditions):
        fail(f"primary analysis: conditions must be keyed by (patience, BN state, threshold) among the {len(CONDITIONS)} conditions")
    for k in (PRIMARY_ORDINARY, PRIMARY_RE_ESTIMATED):
        if k not in conditions:
            fail(f"primary analysis: the condition {k} is missing")
    ordinary = compare_condition(conditions[PRIMARY_ORDINARY])
    re_est = compare_condition(conditions[PRIMARY_RE_ESTIMATED])
    c = compare_conditions(re_est, ordinary)
    rows = []
    for x, y in zip(re_est["prepared"], ordinary["prepared"]):
        dr, do = unit_row(x)["delta"], unit_row(y)["delta"]
        rows.append({"unit": x["unit"], "delta_re_estimated": dr, "delta_ordinary": do, "contrast": {k: diff(dr[k], do[k]) for k in METRICS},
                     "n_frames": x["n_frames"], "frame_ids_sha256": x["frame_ids_sha256"]})
    per_seed = {}
    for seed in sorted({pu["seed"] for pu in ordinary["prepared"]}):
        so = condition_of([pu for pu in ordinary["prepared"] if pu["seed"] == seed])
        sr = condition_of([pu for pu in re_est["prepared"] if pu["seed"] == seed])
        per_seed[seed] = {"gap_ordinary": gap_summary(so), "gap_re_estimated": gap_summary(sr),
                          "contrast": contrast_summary(compare_conditions(sr, so))}
    return {"conditions": {"ordinary": list(PRIMARY_ORDINARY), "re-estimated": list(PRIMARY_RE_ESTIMATED)},
            "definition": "contrast = delta(re-estimated BN) - delta(ordinary BN), delta = metric(ResNet-18) - metric(LivenessNet)",
            "gap_ordinary": gap_summary(ordinary), "gap_re_estimated": gap_summary(re_est), "contrast": contrast_summary(c),
            "rows": rows, "per_seed": per_seed}
