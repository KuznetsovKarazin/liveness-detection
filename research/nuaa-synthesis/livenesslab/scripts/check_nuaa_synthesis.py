"""
Test di scripts/nuaa_synthesis.py su un albero research/ sintetico scritto in una cartella temporanea: nessun dato
reale, nessuna dipendenza oltre la libreria standard. La fixture costruisce le undici sorgenti (più quella facoltativa
del confronto con N1, scritta solo dai test che la usano) con numeri inventati, distinti per ogni sottoinsieme e
coerenti fra loro; ogni test di coerenza guasta una sorgente per volta e verifica che build si fermi con il problema
giusto; i test di costruzione confrontano ogni riga e ogni annesso con la fixture.

Uso (dalla radice del repository del gruppo): python3 -I research/nuaa-synthesis/livenesslab/scripts/check_nuaa_synthesis.py [-v]
"""
import contextlib
import copy
import csv
import hashlib
import importlib.util
import io
import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("nuaa_synthesis", HERE / "nuaa_synthesis.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

CNNS = S.NUAA_CNNS
SEEDS = ["42", "43", "44"]


def metrics(n_bf, n_att, rej, acc, auroc):
    return {"n": n_bf + n_att, "n_bona_fide": n_bf, "n_attack": n_att, "accepted_attacks": acc, "rejected_bona_fide": rej,
            "ties_at_threshold": 0, "threshold": 0.5, "decision": "attack if score_attack > 0.5 (tie = bona fide)",
            "apcer": acc / n_att, "bpcer": rej / n_bf, "acer": (acc / n_att + rej / n_bf) / 2, "auroc": auroc}


def metrics_na(n_bf, n_att, rej, acc):
    """Riga con una classe assente: i tassi della classe assente, ACER e AUROC sono NA (None)."""
    return {"n": n_bf + n_att, "n_bona_fide": n_bf, "n_attack": n_att, "accepted_attacks": acc, "rejected_bona_fide": rej,
            "ties_at_threshold": 0, "threshold": 0.5, "decision": "attack if score_attack > 0.5 (tie = bona fide)",
            "apcer": acc / n_att if n_att else None, "bpcer": rej / n_bf if n_bf else None, "acer": None, "auroc": None}


def f4(v):
    return "%.4f" % v


FOLD_STRATA = ("01", "02", "03")


def fstratum(n_bf, n_att, r05, a05, rdev, adev, auroc, n_subj):
    """Uno strato del run a fold (soglie 0,5 e dev) con i tassi calcolati dai conteggi; una classe assente dà NA."""
    if n_bf == 0:
        r05 = rdev = 0
    if n_att == 0:
        a05 = adev = 0
    if n_bf == 0 or n_att == 0:
        auroc = None
    d = {"n": n_bf + n_att, "n_bona_fide": n_bf, "n_attack": n_att, "n_subjects": n_subj, "auroc": auroc}
    for t, (r, a) in (("05", (r05, a05)), ("dev", (rdev, adev))):
        d["rejected_bona_fide_" + t], d["accepted_attacks_" + t] = r, a
    return refresh(d)


def refresh(d):
    """Ricalcola i tassi di uno strato dai suoi conteggi (per guastare i conteggi senza guastare i tassi)."""
    for t in ("05", "dev"):
        ap = d["accepted_attacks_" + t] / d["n_attack"] if d["n_attack"] else None
        bp = d["rejected_bona_fide_" + t] / d["n_bona_fide"] if d["n_bona_fide"] else None
        d["apcer_" + t], d["bpcer_" + t] = ap, bp
        d["acer_" + t] = (ap + bp) / 2 if ap is not None and bp is not None else None
    d["n"] = d["n_bona_fide"] + d["n_attack"]
    return d


def fsum(parts, n_subj, auroc):
    keys = ("n_bona_fide", "n_attack", "rejected_bona_fide_05", "accepted_attacks_05", "rejected_bona_fide_dev", "accepted_attacks_dev")
    d = {k: sum(p[k] for p in parts) for k in keys}
    d.update({"n_subjects": n_subj, "auroc": auroc})
    return refresh(d)


def summary(values):
    vals = list(values)
    if None in vals:
        return {"values": vals, "median": None, "min": None, "max": None}
    return {"values": vals, "median": sorted(vals)[1], "min": min(vals), "max": max(vals)}


def f6(v):
    return "%.6f" % v


# --- run di ablazione delle sessioni e verifica della pazienza (fixture) ---

ARMS = ("arm_b", "arm_a_star", "arm_b_prime")
ARM_L = {"arm_b": "B", "arm_a_star": "A*", "arm_b_prime": "B'"}
COMPS = (("a_star_minus_b", "arm_a_star", "arm_b", "primary"), ("b_prime_minus_b", "arm_b_prime", "arm_b", "secondary"),
         ("a_star_minus_b_prime", "arm_a_star", "arm_b_prime", "secondary"))
COMP_L = {"a_star_minus_b": "A* - B", "b_prime_minus_b": "B' - B", "a_star_minus_b_prime": "A* - B'"}
KEYS7 = ("apcer_05", "bpcer_05", "acer_05", "apcer_dev", "bpcer_dev", "acer_dev", "auroc")
STRATA4 = ("01", "02", "03", "all")
REASONS = {"zero": "a seed has a zero difference", "sign": "the sign differs across the seeds",
           "interval": "the bootstrap interval of a seed does not exclude 0 on the side of the difference",
           "consistent": "same sign for every seed and every interval excludes 0"}
# indici (ordine braccio, fold, seed) dei modelli costruiti degeneri nella run con pazienza 4; le dieci della fixture sono
# poi ricavate dai dati (degeneri con pazienza 4), così una variante può cambiarle
TEN = (0, 11, 16, 17, 18, 20, 23, 26, 29, 35)
METRIC_NAME = {"apcer_05": "APCER (0.5)", "bpcer_05": "BPCER (0.5)", "acer_05": "ACER (0.5)", "apcer_dev": "APCER (dev threshold)",
               "bpcer_dev": "BPCER (dev threshold)", "acer_dev": "ACER (dev threshold)", "auroc": "AUROC"}
SIDE = ("epochs_run", "restored_epoch", "degenerate", "recovered", "first_recovery", "best_val_loss", "dev_acc_at_checkpoint",
        "dev_auroc_at_checkpoint", "epochs_at_half", "lr_last")
ROWMAP = {"epochs_run": "epochs_run", "restored_epoch": "restored_epoch", "degenerate": "degenerate", "recovered": "recovered",
          "first_recovery": "first_recovery", "best_val_loss": "best_val_loss", "dev_acc": "dev_acc_at_checkpoint",
          "dev_auroc": "dev_auroc_at_checkpoint", "epochs_at_half": "epochs_at_half", "lr_last": "lr_last"}
HIST5 = ("loss", "accuracy", "val_loss", "val_accuracy", "lr")
TOL = 2 ** -20  # tolleranza della fixture: potenza di 2, così uno scarto uguale alla tolleranza è esatto
CRITERIA = {"degenerate": {"restored_epochs": [1, 2], "val_acc_at_best": 0.5, "tolerance": TOL, "text": "restored_epoch in {1, 2} and val_acc_at_best = 0.5"},
            "saturation": {"val_accuracy": 0.5, "tolerance": TOL, "text": "val_accuracy = 0.5"},
            "recovery": {"val_accuracy_min": 0.75, "val_loss_max": 1.0, "text": "first epoch with val_accuracy >= 0.75 and val_loss < 1.0"},
            "checkpoint_agreement_tolerance": 1e-06, "shared_history_difference": 0, "source": "previous package"}


def es_stop(vl, patience, max_epochs=12):
    """Epoche eseguite da un arresto precoce sulla val_loss (miglioramento stretto) con la pazienza data."""
    best, wait = float("inf"), 0
    for e, v in enumerate(vl[:max_epochs], 1):
        if v < best:
            best, wait = v, 0
        else:
            wait += 1
            if wait >= patience:
                return e
    return max_epochs


def history(i):
    """Curve per epoca (12) di un modello sintetico; casi limite: pareggio della val_loss minima (i=3), val_loss 1,0 e
    val_accuracy 0,75 esatte (i=7), lr 6,25e-05 (i=2), accuratezza dev al checkpoint 0,9 esatta e 0,85 (i=11, 16), prima
    ripresa all'epoca 10 esatta (i=17), ripresa che non regge all'ultima epoca (i=35), ripristino all'epoca 2 senza
    degenerazione (i % 8 == 2)."""
    e12 = range(1, 13)
    lr = [0.001 * 0.5 ** ((e - 1) // 4) for e in e12]
    if i in TEN:
        rp = 1 + i % 2
        npp, rn = rp + 4, 12 - i % 3
        va = [0.5] * npp + [0.95] * (12 - npp)
        vl = [2.0 + 0.5 * abs(e - rp) for e in range(1, npp + 1)] + [0.05 + 0.01 * abs(e - rn) for e in range(npp + 1, 13)]
        if i == TEN[-1]:
            va[11] = 0.7  # la ripresa non regge all'ultima epoca
        if i == 0:
            va[0] = 0.5 + TOL  # degenere al limite esatto della tolleranza
        if i == 11:
            va[rn - 1] = 0.9
        if i == 16:
            va[rn - 1] = 0.85
        if i == 17:
            va[npp:9] = [0.6] * (9 - npp)  # prima ripresa all'epoca 10
    elif i % 4 == 0:
        rp = rn = 9 + i % 3
        npp = 12
        va = [0.5, 0.5] + [0.9] * 10
        vl = [0.1 + 0.02 * abs(e - rp) for e in e12]
    elif i % 4 == 1:
        rp, npp, rn = 3, 7, 11
        va = [0.6] * 7 + [0.9] * 5
        vl = [0.5 + 0.05 * abs(e - 3) for e in range(1, 8)] + [0.2 + 0.01 * abs(e - 11) for e in range(8, 13)]
    elif i % 4 == 2:  # metà di questi ripristina l'epoca 2 senza essere degenere (accuratezza dev 0,8), come B' fold 4 seed 42
        rp = rn = 2 if i % 8 == 2 else 4
        npp = rp + 4
        va = [0.8] * npp + [0.6] * (12 - npp)
        vl = [0.3 + 0.05 * abs(e - rp) for e in range(1, npp + 1)] + [1.5] * (12 - npp)
        if i == 2:
            lr[11] = 6.25e-05
    else:
        va = [0.6] * 5 + [0.85] * 7
        vl = [0.4 - 0.02 * e for e in e12]
        if i == 3:  # pareggio: la prima epoca del minimo è quella ripristinata
            vl = [0.4 - 0.02 * e for e in range(1, 9)] + [0.4 - 0.02 * 8] + [0.3] * 3
        if i == 7:  # epoca 2: accuratezza 0,9 ma val_loss 1,0 (non ripresa); epoca 3: accuratezza 0,75 esatta (ripresa)
            va[1], vl[1], va[2], vl[2] = 0.9, 1.0, 0.75, 0.5
    return {"va": va, "vl": vl, "lr": lr, "loss": [0.1 / e for e in e12], "accuracy": [0.9 + e / 1000 for e in e12]}


def side_values(h, n, i, run, changed=None):
    vl, va = h["vl"][:n], h["va"][:n]
    r = vl.index(min(vl)) + 1
    ok = [a >= 0.75 and l < 1.0 for a, l in zip(va, vl)]
    first = ok.index(True) + 1 if True in ok else None
    return {"epochs_run": n, "restored_epoch": r, "degenerate": r in (1, 2) and abs(va[r - 1] - 0.5) <= TOL, "recovered": first is not None,
            "first_recovery": first, "best_val_loss": min(vl), "dev_acc_at_checkpoint": va[r - 1],
            "dev_auroc_at_checkpoint": 0.99 - 0.001 * (i % 5) if run == 0 or not changed else 0.995,
            "epochs_at_half": sum(abs(a - 0.5) <= TOL for a in va), "lr_last": h["lr"][n - 1], "holding_last": ok[-1]}


def rule(ds):
    diffs = [d["difference"] for d in ds]
    if any(x == 0 for x in diffs):
        return "zero"
    if not (all(x > 0 for x in diffs) or all(x < 0 for x in diffs)):
        return "sign"
    if not all((d["low"] > 0) if d["difference"] > 0 else (d["high"] < 0) for d in ds):
        return "interval"
    return "consistent"


def sha_text(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def csv_dump(rows, columns):
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(columns)
    for r in rows:
        w.writerow(["1" if r[c] is True else "0" if r[c] is False else "" if r[c] is None else str(r[c]) for c in columns])
    return out.getvalue()


class Fixture:
    """Le otto sorgenti sintetiche, modificabili prima della scrittura."""

    def __init__(self, overrides=None):
        self.overrides = overrides or {}
        self.mc = {"official_train": (3491, 1743, 1748, 9), "official_test": (9123, 3362, 5761, 16),
                   "cnn_train": (2507, 1345, 1162, 7), "cnn_es": (493, 155, 338, 2), "cnn_test": (3000, 1500, 1500, 16),
                   "m1_dev": (574, 180, 394, 2), "c1": (300, 150, 150, 16)}
        self.c1 = {}
        for i, a in enumerate(CNNS):
            self.c1[a] = {"rej": 100 + i, "acc": 10 + 2 * i, "rej_d": 90 + 3 * i, "acc_d": 15 + i, "auc": 0.6 + i / 100,
                          "eer": 0.3 + i / 100, "thr": 0.5 + i / 10, "sign_changes": i % 2, "n_na": 0,
                          "lo_min": -0.06 - i / 100, "lo_max": -0.01 + i / 100}
        self.cross = {}
        k = 0
        for arch in ("livenessnet", "attacknet_v1", "attacknet_v2_1", "attacknet_v2_2"):
            for src in ("casia_fasd", "celeba_spoof", "synthaspoof"):
                self.cross["%s__%s" % (arch, src)] = (0.30 + k / 100, 0.40 + k / 100, 0.55 - k / 100, 0.20 + k / 100)
                k += 1
        for name in ("livenessnet__casia_fasd-pooled", "attacknet_v2_2__casia_fasd-pooled",
                     "livenessnet__celeba_spoof-pooled", "attacknet_v2_2__celeba_spoof-pooled"):
            self.cross[name] = (0.70 + k / 100, 0.80 + k / 100, 0.20 - k / 100, 0.60 + k / 100)
            k += 1
        self.context = {name: (0.01 * i, 0.02 * i, 0.9 - 0.01 * i, 0.05 + 0.01 * i) for i, name in enumerate(S.CONTEXT_ROWS)}
        self.n1 = {s: {"test_3000": metrics(1500, 1500, 1100 + 10 * i, 100 + 20 * i, 0.40 + i / 100),
                       "c1_300": metrics(150, 150, 111 + i, 12 + i, 0.41 + i / 100)} for i, s in enumerate(SEEDS)}
        self.training = {s: {"epochs_run": 9 + i, "restored_epoch": 5 + i, "best_val_loss": 0.001 * (i + 1),
                             "weights_sha256": ("w%s" % s) * 16} for i, s in enumerate(SEEDS)}
        self.target = {"rows": 491, "n_bona_fide": 243, "n_attack": 248, "n_subjects": 9}
        self.n2 = {s: {"sessions_01_02_491": metrics(243, 248, i, i, 1.0 - i / 100),
                       "sessions_01_02_fitting_subjects": metrics(218, 192, 0, i, 0.99 - i / 100),
                       "sessions_01_02_validation_subjects": metrics(25, 56, i, 0, 1.0),
                       "session_03_9_subjects": metrics(614, 924, 400 + i, 100 + 5 * i, 0.35 + i / 100),
                       "session_03_fitting_subjects": metrics(614, 711, 400 + i, 80 + 4 * i, 0.33 + i / 100),
                       "session_03_validation_subjects": metrics_na(0, 213, 0, 20 + i),
                       "session_03_3000": copy.deepcopy(self.n1[s]["test_3000"])} for i, s in enumerate(SEEDS)}
        self.comb = {"whole_image": metrics(1500, 1500, 1490, 220, 0.22), "face_crop": metrics(1500, 1500, 1220, 350, 0.37)}
        v21 = self.c1["attacknet_v2_1__nuaa"]
        self.crop = {"cache_0ac7143": metrics(150, 150, 120, 5, 0.70),
                     "c1_a80f6f9": metrics(150, 150, v21["rej"], v21["acc"], v21["auc"])}
        self.crop_ref = metrics(150, 150, v21["rej"], v21["acc"], v21["auc"])
        self.paired = {"n": 300, "max_abs_diff": 0.3, "median_abs_diff": 1e-5, "mean_abs_diff": 0.01,
                       "mean_signed_diff_cache_minus_c1": 0.004, "abs_diff_bins": {"zero": 97},
                       "decision_changes_c1_to_cache": {"unchanged": {"all": 291}}, "frames_equal": 0}
        self.n1_run, self.n2s_run, self.n2c_run, self.crop_run = ("20261024-N1-LivenessNet-fb93496", "20261029-N2-sessions-2ac3e29",
                                                                   "20261029-N2-combined-247dfd7", "20261102-C1CROP-f6747cc")
        self.n2s_n1_study = self.n1_run
        self.passed = {"m2": True, "n1": True, "n2s": True, "n2c": True, "crop": True}
        self.dirty = {"m2": False, "n1": False, "n2s": False, "n2c": False, "crop": False, "subj": False}
        self.dev_flag = {a: "" for a in CNNS}
        self.crop_ref_run = S.C1_RUN
        self.subj_reproduced = True
        self.m1_missing_column = None
        self.c1_missing_column = None
        self.c1_missing_row = None
        self.m1_missing_row = None
        self.m1_duplicate_row = None
        self.c1_duplicate_row = None
        self.n2_weights = {s: self.training[s]["weights_sha256"] for s in SEEDS}
        self.n1_threshold = 0.5
        self.n2_seeds = [42, 43, 44]
        self.subj_counts = {a: {"at_0_5": (d["acc"], d["rej"]), "at_m1": (d["acc_d"], d["rej_d"])} for a, d in self.c1.items()}
        self.subj_acer_m1 = {a: (d["acc_d"] / 150 + d["rej_d"] / 150) / 2 for a, d in self.c1.items()}
        self.subj_thr = {a: d["thr"] for a, d in self.c1.items()}
        self.n2_variants = dict(self.comb)
        self.commits = {"n1": "fb9349685dfc15fae99f97706ea86f4eabd67c6b", "subj": "a5fcbbc", "n2s": "2ac3e290c9081454e9d058a294ab84156de96235",
                        "n2c": "247dfd79d565985a4f3133b553fee36a27a4aa26", "crop": "f6747cc9e56657976403a63f6f85a16720650149"}
        self.m2_subjects = {"fitted": 7, "early_stopping": 2}
        self.m2_exclusions = {"n": 491, "n_bona_fide": 243, "n_attack": 248, "n_subjects": 9}
        self.folds_init()
        self.ablation_init()
        self.compare = None  # None: confronto con N1 generato dallo stato corrente della fixture (compare_json)

    # --- run a fold di soggetti ---
    def folds_init(self):
        """Schema, aggregato e record del run a fold, coerenti fra loro e con le liste ufficiali M2 della fixture."""
        self.fold_cells = {"1": {"01": (250, 280), "02": (240, 300), "03": (930, 1230)},
                           "2": {"01": (230, 190), "02": (120, 190), "03": (1020, 1360)},
                           "3": {"01": (280, 180), "02": (250, 200), "03": (860, 1670)},
                           "4": {"01": (678, 194), "02": (247, 200), "03": (0, 1515)}}
        assert sum(c[0] for f in self.fold_cells.values() for c in f.values()) == 1743 + 3362
        assert sum(c[1] for f in self.fold_cells.values() for c in f.values()) == 1748 + 5761
        folds = list(self.fold_cells)
        self.fold_totals = [sum(sum(c) for c in self.fold_cells[k].values()) for k in folds]
        per = {}
        for k in folds:
            fi = int(k)
            per[k] = {}
            for si, s in enumerate(SEEDS):
                blk = {}
                for sti, st in enumerate(FOLD_STRATA):
                    n_bf, n_att = self.fold_cells[k][st]
                    r05, a05 = 5 * fi + 3 * si + 2 * sti + 1, 4 * fi + 2 * si + sti
                    blk[st] = fstratum(n_bf, n_att, r05, a05, r05 + si + 1, a05 + 2 * sti, 0.90 + fi / 100 + si / 1000 + sti / 10000, 3 - sti % 2)
                blk["all"] = fsum([blk[st] for st in FOLD_STRATA], 4, 0.95 + fi / 100 + si / 1000)
                per[k][s] = blk
        pooled, boot = {}, {}
        for si, s in enumerate(SEEDS):
            pooled[s], boot[s] = {}, {}
            for sti, st in enumerate(FOLD_STRATA + ("all",)):
                pooled[s][st] = fsum([per[k][s][st] for k in folds], (9, 9, 16, 16)[sti], 0.93 + si / 100 + sti / 1000)
                m = pooled[s][st]
                boot[s][st] = {"n_subjects": m["n_subjects"], "resamples": 1000, "seed": 0, "percentiles": [2.5, 97.5],
                               "acer_05": {"low": m["acer_05"] / 2, "high": m["acer_05"] * 1.5 + 0.01, "valid_resamples": 1000},
                               "acer_dev": {"low": m["acer_dev"] / 2, "high": m["acer_dev"] * 1.5 + 0.01, "valid_resamples": 1000},
                               "auroc": {"low": m["auroc"] - 0.05, "high": min(1.0, m["auroc"] + 0.01), "valid_resamples": 1000}}
        keys = ("apcer_05", "bpcer_05", "acer_05", "apcer_dev", "bpcer_dev", "acer_dev", "auroc")
        training = {}
        for k in folds:
            for si, s in enumerate(SEEDS):
                fi = int(k)
                training["fold%s_seed%s" % (k, s)] = {
                    "epochs_run": 12, "restored_epoch": 8 + (fi + si) % 5, "restored_by": "early stopping",
                    "best_val_loss": 0.01 * (fi + si), "train_seconds": 700.0 + 10 * fi + si, "n_fitting": 2100, "n_dev": 600,
                    "weights_sha256": "a" * 64, "dev_threshold": 0.1 * fi + 0.01 * si + 0.0012345, "dev_threshold_sha256": "b" * 64,
                    "dev_apcer_at_threshold": 0.01 * fi, "dev_bpcer_at_threshold": 0.01 * fi}
        self.folds_run = "20261103-FOLDSRUN-e174812"
        bootstrap = {"resamples": 1000, "seed": 0, "percentiles": [2.5, 97.5], "min_subjects": 5}
        self.fagg = {"run_id": self.folds_run, "commit": "e174812", "dirty": False, "config_sha256": "7" * 64,
                     "delivered_configuration": True, "command": "run", "finished": "2026-11-04T02:41:00+0100",
                     "bootstrap": copy.deepcopy(bootstrap), "decision": "attack if score_attack > threshold (tie = bona fide)",
                     "strata": ["01", "02", "03", "all"], "per_fold_seed": per,
                     "per_fold_summary_over_seeds": {k: {st: {key: summary(per[k][s][st][key] for s in SEEDS) for key in keys}
                                                         for st in per[k]["42"]} for k in folds},
                     "pooled_per_seed": pooled,
                     "pooled_summary_over_seeds": {st: {key: summary(pooled[s][st][key] for s in SEEDS) for key in keys} for st in pooled["42"]},
                     "bootstrap_per_seed": boot, "training": training}
        self.fsch = {"schema": "nuaa-subject-folds/1", "run_id": "20261103-FOLDS-67b7f3d", "commit": "67b7f3d", "dirty": False,
                     "config_sha256": "d" * 64, "train_seeds": [42, 43, 44], "folds": {"k": 4},
                     "sampling": {"fitting_per_cell": 350, "dev_per_cell": 100}, "per_fold": {}}
        for k in folds:
            fi = int(k)
            counts = {}
            for st in FOLD_STRATA:
                counts["%s/bona_fide" % st] = {"frames": self.fold_cells[k][st][0], "subjects": 2}
                counts["%s/attack" % st] = {"frames": self.fold_cells[k][st][1], "subjects": 3}
            counts["total"] = {"frames": sum(sum(c) for c in self.fold_cells[k].values()), "subjects": 4}
            self.fsch["per_fold"][k] = {
                "held_out_fold": fi, "dev_fold": fi % 4 + 1, "fitting_folds": sorted(set(range(1, 5)) - {fi, fi % 4 + 1}),
                "roles": {"fitting": {"n_subjects": 8, "counts": {"total": {"frames": 2100}}},
                          "dev": {"n_subjects": 4, "counts": {"total": {"frames": 600}}},
                          "test": {"n_subjects": 4, "counts": counts}}}
        self.frec = {"run_id": self.folds_run, "commit": "e174812", "dirty": False, "config_sha256": "7" * 64,
                     "delivered_configuration": True, "command": "run", "finished": "2026-11-04T02:41:00+0100",
                     "wall_seconds": 9703.3, "timings": {"train " + n: {"wall_clock_seconds": 1.0} for n in training},
                     "checks": {"schema": {"run_id": "20261103-FOLDS-67b7f3d", "test_per_fold": list(self.fold_totals)},
                                "schema_config": {"sha256": "d" * 64, "wall_time_cap_minutes": 360},
                                "originals": {"located_and_hashed": 12614}},
                     "cache_check": {"train": {"equal": 3000, "different": 0}, "test": {"equal": 3000, "different": 0}},
                     "extraction": {"n": 6614}, "schema_run": {"run_id": "20261103-FOLDS-67b7f3d", "config_sha256": "d" * 64},
                     "train_seeds": [42, 43, 44], "folds": 4, "models": 12, "bootstrap": copy.deepcopy(bootstrap)}

    # --- run di ablazione delle sessioni (pazienza 4), run con pazienza 12 e confronto delle due ---
    def ablation_init(self):
        folds = list(self.fold_cells)
        self.abl_models = [(a, k, s) for a in ARMS for k in folds for s in SEEDS]
        self.hist, self.sides = {}, {}
        for i, key in enumerate(self.abl_models):
            h = history(i)
            if i in self.overrides:
                self.overrides[i](h)
            h["npp"] = es_stop(h["vl"], 4)
            assert es_stop(h["vl"], 12) == 12
            p = side_values(h, h["npp"], i, 0)
            h["rp"] = p["restored_epoch"]
            h["rn"] = h["vl"].index(min(h["vl"])) + 1
            self.hist[key] = h
            self.sides[key] = (p, side_values(h, 12, i, 1, changed=h["rp"] != h["rn"]))
        self.ten = [i for i, key in enumerate(self.abl_models) if self.sides[key][0]["degenerate"]]
        self.cells = {"arm_b": {"01/bona_fide": 350, "01/attack": 350, "02/bona_fide": 350, "02/attack": 350},
                      "arm_a_star": {"01/bona_fide": 234, "01/attack": 233, "02/bona_fide": 233, "02/attack": 234,
                                     "03/bona_fide": 233, "03/attack": 233},
                      "arm_b_prime": {"01/bona_fide": 234, "01/attack": 233, "02/bona_fide": 233, "02/attack": 234}}
        self.dev_cells = {"01/bona_fide": 100, "01/attack": 100, "02/bona_fide": 100, "02/attack": 100}
        self.abl_ids = [("20261106-SESSABLRUN-d999458", "d999458", "4" * 64, "2026-11-06T11:31:29+0100"),
                        ("20261109-SESSABLRUN-6a50265", "6a50265", "c" * 64, "2026-11-09T19:06:18+0100")]
        self.abl = [self.abl_run(0), self.abl_run(1)]
        self.patver_init()

    def changed(self, key):
        return self.hist[key]["rp"] != self.hist[key]["rn"]

    def abl_run(self, run):
        folds = list(self.fold_cells)
        rid, commit, cfg, fin = self.abl_ids[run]
        per = {a: {k: {} for k in folds} for a in ARMS}
        for ai, a in enumerate(ARMS):
            for k in folds:
                fi = int(k)
                for si, s in enumerate(SEEDS):
                    shift = 7 if run == 1 and self.changed((a, k, s)) else 0
                    blk = {}
                    for sti, st in enumerate(FOLD_STRATA):
                        n_bf, n_att = self.fold_cells[k][st]
                        r05, a05 = 3 * ai + 5 * fi + 2 * si + sti + 1 + shift, 2 * ai + 4 * fi + si + sti + shift
                        blk[st] = fstratum(n_bf, n_att, r05, a05, r05 + 2, a05 + 1 + sti,
                                           0.6 + ai / 10 + fi / 100 + si / 1000 + sti / 10000 + shift / 1e5, 3 - sti % 2)
                    blk["all"] = fsum([blk[st] for st in FOLD_STRATA], 4, 0.7 + ai / 10 + fi / 100 + si / 1000 + shift / 1e5)
                    per[a][k][s] = blk
        pooled, boot = {a: {} for a in ARMS}, {a: {} for a in ARMS}
        for ai, a in enumerate(ARMS):
            for si, s in enumerate(SEEDS):
                pooled[a][s], boot[a][s] = {}, {}
                for sti, st in enumerate(STRATA4):
                    m = fsum([per[a][k][s][st] for k in folds], (9, 9, 16, 16)[sti], 0.65 + (ai if run == 0 else 2 - ai) / 10 + si / 100 + sti / 1000 + run / 1e4)
                    pooled[a][s][st] = m
                    boot[a][s][st] = {"n_subjects": m["n_subjects"], "resamples": 1000, "seed": 0, "percentiles": [2.5, 97.5],
                                      "acer_05": {"low": m["acer_05"] / 2, "high": m["acer_05"] * 1.5 + 0.01, "valid_resamples": 1000},
                                      "acer_dev": {"low": m["acer_dev"] / 3, "high": m["acer_dev"] * 1.5 + 0.02, "valid_resamples": 1000},
                                      "auroc": {"low": m["auroc"] - 0.05, "high": min(1.0, m["auroc"] + 0.01), "valid_resamples": 1000}}
        comps, prim = {}, {}
        for ci, (cname, mi, su, role) in enumerate(COMPS):
            pfs = {k: {s: {st: {m: None if per[mi][k][s][st][m] is None or per[su][k][s][st][m] is None
                                else per[mi][k][s][st][m] - per[su][k][s][st][m] for m in KEYS7} for st in STRATA4} for s in SEEDS}
                   for k in folds}
            pps = {s: {} for s in SEEDS}
            for s in SEEDS:
                for sti, st in enumerate(STRATA4):
                    pps[s][st] = {"n_subjects": pooled[mi][s][st]["n_subjects"]}
                    for mk, m in enumerate(KEYS7):
                        x, y = pooled[mi][s][st][m], pooled[su][s][st][m]
                        d = x - y
                        f = 0.5 if (ci + sti + mk + run) % 3 == 0 or (run == 1 and mk == 6) else 2.0
                        pps[s][st][m] = {"difference": d, "minuend": x, "subtrahend": y, "low": d - abs(d) * f, "high": d + abs(d) * f,
                                         "valid_resamples": 1000}
            summ, verd = {}, {}
            for st in STRATA4:
                summ[st], verd[st] = {}, {}
                for m in KEYS7:
                    ds = [pps[s][st][m] for s in SEEDS]
                    summ[st][m] = summary(d["difference"] for d in ds)
                    reason = rule(ds)
                    ok = reason == "consistent"
                    verd[st][m] = {"verdict": "consistent" if ok else "inconclusive",
                                   "direction": ("%s %s %s than %s" % (ARM_L[mi], "higher" if ds[0]["difference"] > 0 else "lower", METRIC_NAME[m],
                                                                       ARM_L[su])) if ok else None,
                                   "reason": REASONS[reason],
                                   "label": "primary" if cname == "a_star_minus_b" and st == "03" and m in ("auroc", "acer_dev") else "secondary"}
            comps[cname] = {"minuend": mi, "subtrahend": su, "role": role, "per_fold_seed": pfs, "pooled_per_seed": pps,
                            "pooled_summary_over_seeds": summ, "verdicts": verd}
        for m in ("auroc", "acer_dev"):
            prim[m] = dict(comps["a_star_minus_b"]["verdicts"]["03"][m])
        n_fit = {a: sum(self.cells[a].values()) for a in ARMS}
        names, training = [], {}
        for i, (a, k, s) in enumerate(self.abl_models):
            name = "%s_fold%s_seed%s" % (a, k, s)
            names.append(name)
            sv = self.sides[(a, k, s)][run]
            training[name] = {"arm": a, "fold": int(k), "seed": int(s), "epochs_run": sv["epochs_run"], "restored_epoch": sv["restored_epoch"],
                              "restored_by": "early stopping" if sv["epochs_run"] < 12 else "end of epochs (set_weights)",
                              "best_val_loss": sv["best_val_loss"], "train_seconds": 100.0 + i + 200 * run, "n_fitting": n_fit[a],
                              "n_dev": 400, "weights_sha256": "%064x" % (i + 100 * run),
                              "dev_threshold": 0.3 + i / 100 + (0.001 if run == 1 and self.changed((a, k, s)) else 0),
                              "dev_threshold_sha256": "e" * 64, "dev_apcer_at_threshold": 0.01, "dev_bpcer_at_threshold": 0.01}
        bootstrap = {"resamples": 1000, "seed": 0, "percentiles": [2.5, 97.5], "min_subjects": 5, "recipe": "restarted per seed (fixture)"}
        ident = {"run_id": rid, "commit": commit, "dirty": False, "config_sha256": cfg, "delivered_configuration": True, "command": "run"}
        extra = {} if run == 0 else {"configuration_name": "patience12_verification",
                                     "amendment": {"parent_config_sha256": self.abl_ids[0][2], "changed_field": "training.early_stopping_patience",
                                                   "from": 4, "to": 12, "other_fields": "equal to the parent configuration"}}
        agg = dict(ident, **copy.deepcopy(extra), status="PASS", finished=fin, completed=36, trainings=36, missing=[],
                   bootstrap=copy.deepcopy(bootstrap), decision="attack if score_attack > threshold (tie = bona fide)",
                   strata=list(STRATA4), per_arm_fold_seed=per, pooled_per_arm_seed=pooled, bootstrap_per_arm_seed=boot,
                   pooled_summary_over_seeds={a: {st: {key: summary(pooled[a][s][st][key] for s in SEEDS) for key in KEYS7}
                                                  for st in STRATA4} for a in ARMS},
                   comparisons=comps, primary_verdicts=prim, reading_rule="consistent only with the same sign and every interval excluding 0 (fixture)",
                   reference_a={"label": "reference", "run_id": self.folds_run, "aggregate_sha256": None}, training=training)
        rec = dict(ident, **copy.deepcopy(extra), status="PASS", finished=fin, wall_seconds=10000.0 + 7000 * run,
                   timings={"train " + n: {"wall_clock_seconds": 1.0} for n in names}, trainings=36, training_order=list(names),
                   completed=list(names), missing=[], stopped=None, cap_minutes=360, final_writes_after_cap=False,
                   checks={"schema": {"run_id": self.fsch["run_id"], "test_per_fold": list(self.fold_totals)},
                           "schema_config": {"sha256": self.fsch["config_sha256"]},
                           "folds_run": {"run_id": self.folds_run, "aggregate_json_sha256": None, "run_record_json_sha256": None},
                           "cells": {"counts": copy.deepcopy(self.cells), "dev": dict(self.dev_cells)}},
                   train_seeds=[42, 43, 44], bootstrap=copy.deepcopy(bootstrap))
        status = dict(ident, **copy.deepcopy(extra), status="PASS", passed=True, finished=fin, wall_seconds=rec["wall_seconds"],
                      completed=list(names), missing=[], stopped=None, final_writes_after_cap=False, public_sums_sha256=None,
                      restricted_sums_sha256="%064x" % (777 + run))
        return {"agg": agg, "rec": rec, "status": status}

    def patver_init(self):
        labels = ["%s fold %s seed %s" % (ARM_L[a], k, s) for a, k, s in self.abl_models]
        rows, crows, curves = [], [], []
        for i, (a, k, s) in enumerate(self.abl_models):
            h, (P, N) = self.hist[(a, k, s)], self.sides[(a, k, s)]
            in_ten = i in self.ten
            crow = {"arm": a, "fold": k, "seed": s, "in_ten": in_ten, "new_status": "complete", "shared_epochs": h["npp"],
                    "shared_identical": True}
            crow.update({"parent_" + f: P[f] for f in SIDE})
            crow.update({"new_" + f: N[f] for f in SIDE})
            crow.update({"max_abs_diff_shared_" + f: 0.0 for f in HIST5})
            crows.append(crow)
            rows.append({"model": labels[i], "in_ten": in_ten, "new_status": "complete",
                         "parent": {jf: P[cf] for jf, cf in ROWMAP.items()}, "new": {jf: N[cf] for jf, cf in ROWMAP.items()},
                         "shared_epochs": h["npp"], "shared_identical": True, "max_abs_diff_shared": {f: 0.0 for f in HIST5}})
            for e in range(1, 13):
                va, vl = h["va"][e - 1], h["vl"][e - 1]
                curves.append({"arm": a, "fold": k, "seed": s, "in_ten": in_ten, "epoch": e, "loss": h["loss"][e - 1],
                               "accuracy": h["accuracy"][e - 1], "val_loss": vl, "val_accuracy": va, "lr": h["lr"][e - 1],
                               "is_restored": e == N["restored_epoch"], "saturated": abs(va - 0.5) <= TOL, "shared_with_parent": e <= h["npp"],
                               "degenerate": N["degenerate"], "recovered_by_epoch": va >= 0.75 and vl < 1.0})
        self.pv_rows, self.pv_curves = crows, curves
        groups = {"the ten": [i for i in range(36) if i in self.ten], "the other 26": [i for i in range(36) if i not in self.ten],
                  "all 36": list(range(36))}
        msr = {"first_recovery": "first_recovery", "restored_epoch": "restored_epoch", "best_val_loss": "best_val_loss",
               "dev_acc": "dev_acc_at_checkpoint", "dev_auroc": "dev_auroc_at_checkpoint", "epochs_at_half": "epochs_at_half",
               "lr_last": "lr_last"}
        summ = []
        for g, idx in groups.items():
            for run, side in ((0, "parent"), (1, "new")):
                e = {"group": g, "side": side}
                for jm, cf in msr.items():
                    vals = [self.sides[self.abl_models[i]][run][cf] for i in idx if self.sides[self.abl_models[i]][run][cf] is not None]
                    e[jm] = {"models": len(vals), "median": statistics.median(vals) if vals else None,
                             "min": min(vals) if vals else None, "max": max(vals) if vals else None}
                summ.append(e)
        vp, vn = self.abl[0]["agg"]["comparisons"], self.abl[1]["agg"]["comparisons"]
        drows, changed = [], 0
        for c, _, _, _ in COMPS:
            for st in STRATA4:
                for m in KEYS7:
                    a, b = vp[c]["verdicts"][st][m], vn[c]["verdicts"][st][m]
                    drows.append({"comparison": COMP_L[c], "stratum": st, "metric": m,
                                  "parent": {"verdict": a["verdict"], "direction": a["direction"], "reason": a["reason"],
                                             "median": vp[c]["pooled_summary_over_seeds"][st][m]["median"]},
                                  "new": {"verdict": b["verdict"], "direction": b["direction"], "reason": b["reason"],
                                          "median": vn[c]["pooled_summary_over_seeds"][st][m]["median"]}})
                    changed += (a["verdict"], a["direction"]) != (b["verdict"], b["direction"])
        self.verdicts_changed = changed
        per_arm = []
        for run, side in ((0, "parent"), (1, "new")):
            for a in ARMS:
                cards = [self.abl[run]["agg"]["training"]["%s_fold%s_seed%s" % (a, k, s)] for k in self.fold_cells for s in SEEDS]
                ep, secs = sum(t["epochs_run"] for t in cards), sum(t["train_seconds"] for t in cards)
                per_arm.append({"run": side, "arm": ARM_L[a], "models": 12, "epochs": ep, "train_seconds": secs, "seconds_per_epoch": secs / ep})
        recs = [self.abl[0]["rec"], self.abl[1]["rec"]]
        inputs = {}
        for key, run, p in (("parent_run", 0, 4), ("verification_run", 1, 12)):
            rid, commit, cfg, _ = self.abl_ids[run]
            inputs[key] = {"run_id": rid, "status": "PASS", "config_sha256": cfg, "commit": commit, "status_sha256": None,
                           "public_sums_sha256": None, "restricted_sums_sha256": None, "early_stopping_patience": p,
                           "complete_models": 36, "missing_models": 0}
        ten = [labels[i] for i in self.ten]
        sides = [self.sides[m] for m in self.abl_models]
        ten_rec = [labels[i] for i in self.ten if sides[i][1]["recovered"]]
        new_deg = [labels[i] for i in range(36) if sides[i][1]["degenerate"]]
        verdict = "positive" if len(ten_rec) == len(ten) and not new_deg else "negative"
        self.pv = {"run_id": "20261109-PATVER-4f2b925", "commit": "4f2b925", "dirty": False, "config_sha256": "f" * 64,
                   "delivered_configuration": True, "criteria": copy.deepcopy(CRITERIA),
                   "recipe": {"max_epochs": 12, "monitor": "val_loss"}, "original_degenerate": sorted(ten),
                   "models_without_recovery_parent": [labels[i] for i in range(36) if i not in self.ten and not sides[i][0]["recovered"]],
                   "inputs": inputs, "verdict": verdict,
                   "sections": {"a": {"rule": "shared histories", "models_compared": 36, "identical": 36, "valid": True, "largest_difference": None,
                                      "new_history_shorter_than_shared": [], "max_abs_difference_per_field": {f: 0.0 for f in HIST5}},
                                "b": {"rule": "primary outcome", "verdict": verdict, "conclusion": "From the rows.", "verification_status": "PASS",
                                      "ten": sorted(ten), "ten_recovered": len(ten_rec), "ten_complete": len(ten),
                                      "ten_without_recovery": [l for l in ten if l not in ten_rec], "ten_degenerate": [l for l in ten if l in new_deg],
                                      "degenerate_among_complete": new_deg,
                                      "without_recovery_outside_ten": [labels[i] for i in range(36) if i not in self.ten and not sides[i][1]["recovered"]],
                                      "without_recovery_parent": [labels[i] for i in range(36) if not self.sides[self.abl_models[i]][0]["recovered"]],
                                      "missing_models": [], "ten_first_recovery": {labels[i]: sides[i][1]["first_recovery"] for i in self.ten}},
                                "c": {"rule": "secondary outcomes", "summary": summ,
                                      "recovered": {"parent": sum(sd[0]["recovered"] for sd in sides), "new": sum(sd[1]["recovered"] for sd in sides),
                                                    "new_complete": 36}},
                                "d": {"label": "not used to choose or judge the recipe", "rule": "the test", "reported": True,
                                      "source": "public aggregates",
                                      "primary": [{"metric": m, "parent": {k: vp["a_star_minus_b"]["verdicts"]["03"][m][k] for k in ("verdict", "direction", "reason")},
                                                   "new": {k: vn["a_star_minus_b"]["verdicts"]["03"][m][k] for k in ("verdict", "direction", "reason")}}
                                                  for m in ("auroc", "acer_dev")],
                                      "rows": drows, "verdicts_changed": changed},
                                "e": {"rule": "times", "per_arm": per_arm, "new_train_seconds": sum(x["train_seconds"] for x in per_arm[3:]),
                                      "new_wall_seconds": recs[1]["wall_seconds"], "parent_wall_seconds": recs[0]["wall_seconds"],
                                      "estimate_seconds": 13090, "cap_seconds": 21600, "wall_minus_estimate_seconds": 3910.0, "within_cap": True,
                                      "final_writes_after_cap": False, "stopped": None}},
                   "rows": rows, "declarations": {k: True for k in ("no_training", "no_inference", "no_test_scores_read",
                                                                     "test_from_public_aggregate_only", "test_not_used_to_judge_the_recipe",
                                                                     "reading_rule_fixed_before_the_run")}}
        self.pv_status = {"run_id": "20261109-PATVER-4f2b925", "commit": "4f2b925", "dirty": False, "config_sha256": "f" * 64,
                          "delivered_configuration": True, "command": "run", "status": "PASS", "passed": True, "verdict": verdict,
                          "verification_status": "PASS", "public_sums_sha256": None}

    def ablation_files(self, out):
        """Le tredici sorgenti nuove, con le ancore (SHA-256) calcolate sui testi scritti, salvo quelle già fissate da un test."""
        fa, fr = out[S.SOURCES["folds_aggregate"]], out[S.SOURCES["folds_record"]]
        anchors = {}
        for run, pre, base in ((0, "sessabl", "nuaa-session-ablation-run"), (1, "patience", "nuaa-patience-verification")):
            d = copy.deepcopy(self.abl[run])
            if d["agg"]["reference_a"].get("aggregate_sha256") is None:
                d["agg"]["reference_a"]["aggregate_sha256"] = sha_text(fa)
            frun = d["rec"]["checks"]["folds_run"]
            if frun.get("aggregate_json_sha256") is None:
                frun["aggregate_json_sha256"] = sha_text(fa)
            if frun.get("run_record_json_sha256") is None:
                frun["run_record_json_sha256"] = sha_text(fr)
            agg_t, rec_t = json.dumps(d["agg"], indent=1), json.dumps(d["rec"], indent=1)
            sums_t = "%s  aggregate.json\n%s  aggregate.md\n%s  run_record.json\n" % (sha_text(agg_t), "0" * 64, sha_text(rec_t))
            if d["status"].get("public_sums_sha256") is None:
                d["status"]["public_sums_sha256"] = sha_text(sums_t)
            st_t = json.dumps(d["status"], indent=1)
            out[S.SOURCES[pre + "_aggregate"]], out[S.SOURCES[pre + "_record"]] = agg_t, rec_t
            out[S.SOURCES[pre + "_sums"]], out[S.SOURCES[pre + "_status"]] = sums_t, st_t
            anchors[run] = {"status_sha256": sha_text(st_t), "public_sums_sha256": sha_text(sums_t),
                            "restricted_sums_sha256": d["status"].get("restricted_sums_sha256")}
        pv = copy.deepcopy(self.pv)
        for key, run in (("parent_run", 0), ("verification_run", 1)):
            for f, v in anchors[run].items():
                if pv["inputs"][key].get(f) is None:
                    pv["inputs"][key][f] = v
        pv_t = json.dumps(pv, indent=1)
        csv_t, cur_t = csv_dump(self.pv_rows, S.CMP_COLUMNS), csv_dump(self.pv_curves, S.CURVE_COLUMNS)
        sums_t = "%s  aggregate.json\n%s  aggregate.md\n%s  comparison.csv\n%s  curves.csv\n" % (sha_text(pv_t), "0" * 64, sha_text(csv_t), sha_text(cur_t))
        pst = copy.deepcopy(self.pv_status)
        if pst.get("public_sums_sha256") is None:
            pst["public_sums_sha256"] = sha_text(sums_t)
        out[S.SOURCES["patver_aggregate"]], out[S.SOURCES["patver_comparison"]] = pv_t, csv_t
        out[S.SOURCES["patver_curves"]], out[S.SOURCES["patver_sums"]] = cur_t, sums_t
        out[S.SOURCES["patver_status"]] = json.dumps(pst, indent=1)
        return out

    def compare_json(self) -> dict:
        """Confronto con N1 sugli stessi 3.000 frame, nello schema dell'aggregato pubblico del pacchetto di confronto."""
        per, boot, cv = {}, {}, {"acer_05": [], "acer_dev": [], "auroc": []}
        for si, s in enumerate(SEEDS):
            f = fstratum(1500, 1500, 20 + si, 15 + si, 18 + si, 16 + si, 0.94 + si / 100, 16)
            f.pop("n_subjects")
            n = self.n1[s]["test_3000"]
            n1 = {"n": n["n_bona_fide"] + n["n_attack"], "n_bona_fide": n["n_bona_fide"], "n_attack": n["n_attack"],
                  "accepted_attacks_05": n["accepted_attacks"], "rejected_bona_fide_05": n["rejected_bona_fide"],
                  "apcer_05": n["apcer"], "bpcer_05": n["bpcer"], "acer_05": n["acer"], "auroc": n["auroc"]}
            paired = {}
            for t in ("05", "dev"):
                pt, changed = {}, 0
                for cls, nk, fk, n1k in (("bona_fide", "n_bona_fide", "rejected_bona_fide_", "rejected_bona_fide_05"),
                                         ("attack", "n_attack", "accepted_attacks_", "accepted_attacks_05")):
                    fc, nc = f[nk] - f[fk + t], n1[nk] - n1[n1k]
                    bc = nc - 10
                    pt[cls] = {"both_correct": bc, "n1_only_correct": 10, "folds_only_correct": fc - bc, "both_wrong": f[nk] - fc - 10}
                    changed += 10 + fc - bc
                pt.update({"decisions_changed": changed, "n": 3000})
                paired[t] = pt
            subj = {t: {"error_rate": {"folds_lower": 12, "equal": 0, "folds_higher": 4, "not_defined": 0},
                        "acer": {"folds_lower": 5, "equal": 0, "folds_higher": 3, "not_defined": 8}} for t in ("05", "dev")}
            per[s] = {"folds": f, "n1": n1,
                      "difference_folds_minus_n1": {"diff_acer_05": f["acer_05"] - n1["acer_05"],
                                                    "diff_acer_dev": f["acer_dev"] - n1["acer_05"], "diff_auroc": f["auroc"] - n1["auroc"]},
                      "paired_decisions": paired,
                      "per_subject_counts": {"n_subjects": 16, "with_both_classes": 8, "comparison": subj},
                      "test_subjects_per_fold": {"1": 4, "2": 4, "3": 4, "4": 4}}
            boot[s] = {"n_subjects": 16, "resamples": 1000, "seed": 0, "percentiles": [2.5, 97.5]}
            for k in S.COMPARE_BOOT_KEYS:
                boot[s][k] = {"low": -0.5 + si / 100, "high": 0.9 - si / 100, "valid_resamples": 1000}
            cv["acer_05"].append(f["acer_05"]); cv["acer_dev"].append(f["acer_dev"]); cv["auroc"].append(f["auroc"])

        def value(s, key):
            side, rest = key.split("_", 1)
            return per[s]["difference_folds_minus_n1"][key] if side == "diff" else per[s][side][rest]
        return {"run_id": "20261104-FOLDSCOMPARE-abc1234", "commit": "abc1234", "dirty": False, "config_sha256": "c" * 64,
                "delivered_configuration": True, "command": "run", "finished": "2026-11-04T18:00:00+0100",
                "request": {"date": "2026-11-04"}, "task": "comparison",
                "declarations": {k: True for k in S.COMPARE_DECLARATIONS}, "decision": "attack if score_attack > threshold (tie = bona fide)",
                "bootstrap": {"resamples": 1000, "seed": 0, "percentiles": [2.5, 97.5], "min_subjects": 5,
                              "recipe": "generator restarted for every seed; same resamples for both experiments"}, "metrics": {},
                "inputs": {"manifest": {"name": "cnn_test", "sha256": "0" * 64, "rows": 3000, "n_bona_fide": 1500, "n_attack": 1500,
                                        "n_subjects": 16, "session": "03", "schema_config_sha256": "d" * 64},
                           "folds_run": {"run_id": self.folds_run, "rows_per_seed": {s: 12614 for s in SEEDS},
                                         "cnn_test_rows_per_seed": {s: 3000 for s in SEEDS},
                                         "test_subjects_per_fold": {"1": 4, "2": 4, "3": 4, "4": 4}},
                           "n1_run": {"run_id": self.n1_run, "rows_per_seed": {s: 3000 for s in SEEDS}}},
                "code_sha256": {}, "per_seed": per,
                "summary_over_seeds": {k: summary(value(s, k) for s in SEEDS) for k in S.COMPARE_SUMMARY_KEYS},
                "bootstrap_per_seed": boot,
                "control_values": {"source": "coordinator (fixture)",
                                   "control": {"acer_05_percent": [round(100 * v, 2) for v in cv["acer_05"]],
                                               "acer_dev_percent": [round(100 * v, 2) for v in cv["acer_dev"]],
                                               "auroc": [round(v, 5) for v in cv["auroc"]]},
                                   "computed": cv, "comparison": "agreement at the digits given by the coordinator",
                                   "reproduced": True}}

    # --- testi ---
    def c1_table(self) -> str:
        cols = ["analyzer", "n", "n_bona_fide", "n_attack", "apcer", "bpcer", "acer", "eer", "bpcer_at_apcer10", "auc", "accuracy", "n_errors", "n_noface"]
        if self.c1_missing_column:
            cols = [c for c in cols if c != self.c1_missing_column]
        head = "# Evaluation `nuaa`\n\nText.\n\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
        rows = []

        def line(a, ap, bp, eer, au):
            v = {"analyzer": a, "n": 300, "n_bona_fide": 150, "n_attack": 150, "apcer": f4(ap), "bpcer": f4(bp),
                 "acer": f4((ap + bp) / 2), "eer": f4(eer), "bpcer_at_apcer10": "0.5000", "auc": f4(au), "accuracy": "0.6000",
                 "n_errors": 0, "n_noface": 0}
            return "| " + " | ".join(str(v[c]) for c in cols) + " |"
        for a, d in self.c1.items():
            if a == self.c1_missing_row:
                continue
            rows.append(line(a, d["acc"] / 150, d["rej"] / 150, d["eer"], d["auc"]))
        for a, (ap, bp, au, eer) in self.cross.items():
            rows.append(line(a, ap, bp, eer, au))
        for a, (ap, bp, au, eer) in self.context.items():
            rows.append(line(a, ap, bp, eer, au))
            if a == self.c1_duplicate_row:
                rows.append(line(a, ap + 0.1, bp, eer, au))
        rows.append("| livenessnet__nuaa-pooled | 0 | not evaluated here |")
        rows.append("| facemesh | 0 | no score |")
        return head + "\n".join(rows) + "\n"

    def m1_row(self, a, d, cols, extra_dev_rej=0):
        rej_d = d["rej_d"] + extra_dev_rej
        v = {"analyzer": a, "n": 300, "n_bona_fide": 150, "n_attack": 150, "APCER@0.5": f4(d["acc"] / 150),
             "BPCER@0.5": f4(d["rej"] / 150), "ACER@0.5": f4((d["acc"] / 150 + d["rej"] / 150) / 2),
             "accepted attacks@0.5": d["acc"], "rejected bona fide@0.5": d["rej"], "dev threshold": f6(d["thr"]),
             "APCER@dev": f4(d["acc_d"] / 150), "BPCER@dev": f4(rej_d / 150),
             "ACER@dev": f4((d["acc_d"] / 150 + rej_d / 150) / 2), "accepted attacks@dev": d["acc_d"],
             "rejected bona fide@dev": rej_d, "test EER": f4(d["eer"]), "test AUC": f4(d["auc"]),
             "operational dev EER": "0.0000", "dev AUC": "1.0000", "dev flag": self.dev_flag[a]}
        return "| " + " | ".join(str(v[c]) for c in cols) + " |\n"

    def m1_table(self) -> str:
        cols = ["analyzer", "n", "n_bona_fide", "n_attack", "APCER@0.5", "BPCER@0.5", "ACER@0.5", "accepted attacks@0.5",
                "rejected bona fide@0.5", "dev threshold", "APCER@dev", "BPCER@dev", "ACER@dev", "accepted attacks@dev",
                "rejected bona fide@dev", "test EER", "test AUC", "operational dev EER", "dev AUC", "dev flag"]
        if self.m1_missing_column:
            cols = [c for c in cols if c != self.m1_missing_column]
        out = "# M1 table\n\nText.\n\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
        for a, d in self.c1.items():
            if a == self.m1_missing_row:
                continue
            out += self.m1_row(a, d, cols)
            if a == self.m1_duplicate_row:
                out += self.m1_row(a, d, cols, extra_dev_rej=30)
        out += "\n## Sensitivity\n\n| analyzer | n |\n|---|---|\n| livenessnet__nuaa | 278 |\n"
        return out

    def m2_json(self) -> dict:
        return {"schema": "nuaa-m2/1", "passed": self.passed["m2"], "dirty": self.dirty["m2"], "commit": "5fcc1f8314530924099be2d62010734a1f38394d",
                "fitted_subjects": {"n": self.m2_subjects["fitted"]}, "early_stopping_subjects": {"n": self.m2_subjects["early_stopping"]},
                "exclusions": {"official_train_not_sampled": dict(self.m2_exclusions)},
                "manifests": {name: {"file": "manifests/%s.csv" % name, "sha256": "0" * 64, "n": n, "n_bona_fide": bf,
                                     "n_attack": att, "n_subjects": subj} for name, (n, bf, att, subj) in self.mc.items()}}

    def n1_json(self) -> dict:
        return {"passed": self.passed["n1"], "dirty_at_check": self.dirty["n1"], "step": "aggregate", "run_id": self.n1_run, "commit": self.commits["n1"],
                "train_seeds": [42, 43, 44], "split_seed": 42, "threshold": self.n1_threshold,
                "seeds": {s: {"metrics": copy.deepcopy(self.n1[s]), "training": copy.deepcopy(self.training[s])} for s in SEEDS}}

    def n1_subjects_json(self) -> dict:
        out = {"run_id": "20261024-N1-subjects-a5fcbbc", "commit": self.commits["subj"], "dirty": self.dirty["subj"],
               "accepted_m1_table_reproduced": {"result": self.subj_reproduced}, "analyzers": {}, "limits": []}
        for a, d in self.c1.items():
            fa0, fr0 = self.subj_counts[a]["at_0_5"]
            fa1, fr1 = self.subj_counts[a]["at_m1"]
            acer0 = (fa0 / 150 + fr0 / 150) / 2
            acer1 = self.subj_acer_m1[a]
            out["analyzers"][a] = {
                "m1_threshold": self.subj_thr[a],
                "full": {"n_bona_fide": 150, "n_attack": 150,
                         "at_0_5": {"fa": fa0, "fr": fr0, "acer": f6(acer0)},
                         "at_m1": {"fa": fa1, "fr": fr1, "acer": f6(acer1)},
                         "delta_acer": f6(acer1 - acer0)},
                "leave_one_subject_out": {"delta_min": f6(d["lo_min"]), "delta_max": f6(d["lo_max"]),
                                          "n_sign_changes": d["sign_changes"], "n_exclusions": 16, "n_na": d["n_na"]}}
        return out

    def n2s_json(self) -> dict:
        return {"passed": self.passed["n2s"], "dirty_at_check": self.dirty["n2s"], "step": "aggregate", "run_id": self.n2s_run, "commit": self.commits["n2s"],
                "train_seeds": self.n2_seeds, "n1_study": self.n2s_n1_study, "target": dict(self.target),
                "seeds": {s: {"metrics": copy.deepcopy(self.n2[s]), "weights_sha256": self.n2_weights[s]} for s in SEEDS}}

    def n2c_json(self) -> dict:
        return {"passed": self.passed["n2c"], "dirty_at_check": self.dirty["n2c"], "step": "aggregate", "run_id": self.n2c_run, "commit": self.commits["n2c"],
                "model": {"sha256": "3dd0c853" + "0" * 56},
                "variants": {v: {"metrics": {"test_3000": copy.deepcopy(m)}} for v, m in self.n2_variants.items()},
                "n1_reference": {"seeds": {s: {"test_3000": {k: self.n1[s]["test_3000"][k] for k in S.METRIC_FIELDS}} for s in SEEDS}}}

    def crop_json(self) -> dict:
        return {"passed": self.passed["crop"], "dirty_at_check": self.dirty["crop"], "step": "aggregate", "run_id": self.crop_run, "commit": self.commits["crop"],
                "weights": {"sha256": "67d9138d" + "0" * 56}, "metrics": copy.deepcopy(self.crop),
                "paired": copy.deepcopy(self.paired),
                "c1_reference": {"run_id": self.crop_ref_run, "metrics": copy.deepcopy(self.crop_ref)}}

    def files(self) -> dict:
        out = {
            S.SOURCES["c1_table"]: self.c1_table(),
            S.SOURCES["m1_table"]: self.m1_table(),
            S.SOURCES["m2_manifests"]: json.dumps(self.m2_json(), indent=1),
            S.SOURCES["n1_aggregate"]: json.dumps(self.n1_json(), indent=1),
            S.SOURCES["n1_subjects"]: json.dumps(self.n1_subjects_json(), indent=1),
            S.SOURCES["n2_sessions"]: json.dumps(self.n2s_json(), indent=1),
            S.SOURCES["n2_combined"]: json.dumps(self.n2c_json(), indent=1),
            S.SOURCES["crop"]: json.dumps(self.crop_json(), indent=1),
            S.SOURCES["folds_schema"]: json.dumps(self.fsch, indent=1),
            S.SOURCES["folds_aggregate"]: json.dumps(self.fagg, indent=1),
            S.SOURCES["folds_record"]: json.dumps(self.frec, indent=1),
        }
        out[S.SOURCES["folds_n1_compare"]] = json.dumps(self.compare if self.compare is not None else self.compare_json(), indent=1)
        return self.ablation_files(out)

    def write(self, root: Path) -> Path:
        research = root / "research"
        for rel, text in self.files().items():
            p = research / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        return research


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.fx = Fixture()

    def tearDown(self):
        self.tmp.cleanup()

    def research(self, fx=None) -> Path:
        return (fx or self.fx).write(self.root)

    def build_fails(self, fx, needle: str, research=None):
        research = research or fx.write(self.root)
        with self.assertRaises(S.SynthesisError) as cm:
            S.build(research)
        self.assertIn(needle, str(cm.exception))

    def edit_json(self, key: str, fn):
        """Scrive la fixture, poi applica fn al JSON della sorgente key e lo riscrive; torna la cartella research."""
        research = self.fx.write(self.root)
        p = research / S.SOURCES[key]
        d = json.loads(p.read_text(encoding="utf-8"))
        fn(d)
        p.write_text(json.dumps(d), encoding="utf-8")
        return research

    def rows_of(self, doc, block):
        return [r for r in doc["rows"] if r["block"] == block]


class TestBuild(Base):
    def test_rows_and_annexes(self):
        doc = S.build(self.research())
        self.assertEqual(len(doc["rows"]), 57)
        blocks = [r["block"] for r in doc["rows"]]
        self.assertEqual(blocks, ["C1"] * 4 + ["M1"] * 4 + ["N1"] * 3 + ["N2 sessions"] * 6 + ["N2 combined"] * 2 + ["C1 crop check"] * 2
                         + [S.FOLDS_BLOCK] * 15 + [S.COMPARE_BLOCK] * 3 + ["Session ablation run, patience 4"] * 9
                         + ["Session ablation run, patience 12"] * 9)
        self.assertEqual(len(doc["cross_dataset_rows"]), 16)
        self.assertEqual(len(doc["context_rows"]), 10)
        self.assertEqual(len(doc["subject_sensitivity"]), 4)
        self.assertEqual(len(set(r["row"] for r in doc["rows"])), 57)
        for r in doc["rows"]:
            if r["block"] in (S.FOLDS_BLOCK, S.COMPARE_BLOCK) or S.is_abl(r):
                filled = [c for c in S.COLUMNS if c != "bootstrap" or not r["row"].startswith("FOLDS-f")]
                self.assertTrue(all(r[c] != "" for c in filled), r)
            else:
                self.assertTrue(all(r[c] != "" for c in S.COLUMNS if c not in S.FOLDS_ONLY_COLUMNS), r)
                self.assertTrue(all(r[c] == "" for c in S.FOLDS_ONLY_COLUMNS), r)
            self.assertEqual(r["unit"], "image (frame)")
            self.assertIsInstance(r["n_bona_fide"], int)
            self.assertIsInstance(r["rejected_bona_fide"], int)
            self.assertIsInstance(r["accepted_attacks"], int)
        self.assertEqual(doc["manifest_counts"]["cnn_train"]["n"], 2507)
        self.assertEqual(doc["texts"]["train"], "NUAA official train, sessions 01-02: 2,507 fitting frames (7 subjects)")
        self.assertEqual(doc["texts"]["target_491"], "491 frames of the official training list outside fitting and early stopping (243 + 248), sessions 01-02, 9 subjects")

    def test_c1_and_m1_rows_cell_by_cell(self):
        doc = S.build(self.research())
        for i, a in enumerate(CNNS):
            d = self.fx.c1[a]
            c = self.rows_of(doc, "C1")[i]
            self.assertEqual(c["row"], "C1-" + a)
            self.assertEqual(c["model"], S.CNN_NAMES[a] + " (C1 checkpoint of 2026-09-12)")
            self.assertEqual((c["n_bona_fide"], c["n_attack"], c["rejected_bona_fide"], c["accepted_attacks"]), (150, 150, d["rej"], d["acc"]))
            self.assertEqual((c["apcer"], c["bpcer"], c["acer"], c["auroc"]),
                             (f4(d["acc"] / 150), f4(d["rej"] / 150), f4((d["acc"] / 150 + d["rej"] / 150) / 2), f4(d["auc"])))
            self.assertEqual((c["threshold"], c["threshold_origin"], c["run_id"], c["transfer"], c["extractor"], c["preprocessing"]),
                             ("0.5", S.THR_FIXED, S.C1_RUN, S.TRANSFER_SESSION_ALL, S.EXTRACTOR_C1, S.PREP_REF))
            self.assertEqual(c["test_set"], "C1 300 (150 + 150), session 03, 16 subjects")
            self.assertTrue(c["source"].startswith(S.SOURCES["m1_table"]) and S.SOURCES["c1_table"] in c["source"])
            m = self.rows_of(doc, "M1")[i]
            self.assertEqual(m["row"], "M1-" + a)
            self.assertEqual((m["rejected_bona_fide"], m["accepted_attacks"]), (d["rej_d"], d["acc_d"]))
            self.assertEqual((m["apcer"], m["bpcer"], m["acer"], m["auroc"]),
                             (f4(d["acc_d"] / 150), f4(d["rej_d"] / 150), f4((d["acc_d"] / 150 + d["rej_d"] / 150) / 2), f4(d["auc"])))
            self.assertEqual(m["threshold"], f6(d["thr"]))
            self.assertTrue(m["threshold_origin"].startswith("EER on the source dev (574 frames of the 2 early-stopping subjects"))
            self.assertIn("dev AUC 1.0000, operational dev EER 0.0000", m["threshold_origin"])
            self.assertEqual((m["run_id"], m["transfer"], m["source"].split(" (")[0]), (S.M1_RUN, S.TRANSFER_SESSION_ALL, S.SOURCES["m1_table"]))

    def test_n1_rows_cell_by_cell(self):
        doc = S.build(self.research())
        for i, s in enumerate(SEEDS):
            r = self.rows_of(doc, "N1")[i]
            m, tr = self.fx.n1[s]["test_3000"], self.fx.training[s]
            self.assertEqual(r["row"], "N1-seed" + s)
            self.assertEqual(r["model"], "LivenessNet, new training, seed %s (CPU; restored epoch %d of %d, min val_loss %.6f)" % (s, tr["restored_epoch"], tr["epochs_run"], tr["best_val_loss"]))
            self.assertEqual((r["n_bona_fide"], r["n_attack"], r["rejected_bona_fide"], r["accepted_attacks"]), (1500, 1500, m["rejected_bona_fide"], m["accepted_attacks"]))
            self.assertEqual((r["apcer"], r["bpcer"], r["acer"], r["auroc"]), tuple(f4(m[k]) for k in S.RATES))
            self.assertEqual((r["run_id"], r["transfer"], r["extractor"], r["test_set"]),
                             (self.fx.n1_run, S.TRANSFER_SESSION_ALL, S.EXTRACTOR_CACHE + " (cache frames)", "nuaa_cnn_test 3,000 (1,500 + 1,500), session 03, 16 subjects"))
            self.assertEqual(r["source"].split(" (")[0], S.SOURCES["n1_aggregate"])
            self.assertEqual(doc["n1_c1_subset"][s], {k: self.fx.n1[s]["c1_300"][k] for k in S.METRIC_FIELDS})

    def test_n2_session_rows_cell_by_cell(self):
        doc = S.build(self.research())
        rows = self.rows_of(doc, "N2 sessions")
        for i, s in enumerate(SEEDS):
            a, b = self.fx.n2[s]["sessions_01_02_491"], self.fx.n2[s]["session_03_9_subjects"]
            z, t = rows[2 * i], rows[2 * i + 1]
            self.assertEqual((z["row"], t["row"]), ("N2S-01-02-seed" + s, "N2S-03-seed" + s))
            self.assertEqual((z["n_bona_fide"], z["n_attack"], z["rejected_bona_fide"], z["accepted_attacks"]), (243, 248, a["rejected_bona_fide"], a["accepted_attacks"]))
            self.assertEqual((z["apcer"], z["bpcer"], z["acer"], z["auroc"]), tuple(f4(a[k]) for k in S.RATES))
            self.assertEqual((t["n_bona_fide"], t["n_attack"], t["rejected_bona_fide"], t["accepted_attacks"]), (614, 924, b["rejected_bona_fide"], b["accepted_attacks"]))
            self.assertEqual((t["apcer"], t["bpcer"], t["acer"], t["auroc"]), tuple(f4(b[k]) for k in S.RATES))
            self.assertEqual((z["transfer"], t["transfer"]), (S.TRANSFER_SAME, S.TRANSFER_SESSION_9))
            self.assertEqual((z["run_id"], t["run_id"]), (self.fx.n2s_run, self.fx.n2s_run))
            self.assertEqual(z["test_set"], doc["texts"]["target_491"])
            self.assertEqual(t["test_set"], "session 03 test frames of the same 9 training subjects (614 + 924), inside nuaa_cnn_test")
            self.assertEqual((z["extractor"], t["extractor"]), (S.EXTRACTOR_CACHE + " (frozen copy)", S.EXTRACTOR_CACHE + " (cache frames)"))
            self.assertEqual(z["source"].split(" (")[0], S.SOURCES["n2_sessions"])
            for k in ("sessions_01_02_fitting_subjects", "sessions_01_02_validation_subjects", "session_03_fitting_subjects", "session_03_3000"):
                self.assertEqual(doc["n2_sessions_subsets"][k][s], {kk: self.fx.n2[s][k][kk] for kk in S.METRIC_FIELDS})
            self.assertEqual(doc["texts"]["s03_9"], t["test_set"])

    def test_combined_and_crop_rows_cell_by_cell(self):
        doc = S.build(self.research())
        w, c = self.rows_of(doc, "N2 combined")
        self.assertEqual((w["row"], c["row"]), ("N2C-whole_image", "N2C-face_crop"))
        for r, v in ((w, "whole_image"), (c, "face_crop")):
            m = self.fx.comb[v]
            self.assertEqual((r["rejected_bona_fide"], r["accepted_attacks"]), (m["rejected_bona_fide"], m["accepted_attacks"]))
            self.assertEqual((r["apcer"], r["bpcer"], r["acer"], r["auroc"]), tuple(f4(m[k]) for k in S.RATES))
            self.assertEqual((r["extractor"], r["preprocessing"], r["transfer"], r["training_data"], r["run_id"]),
                             (S.VARIANTS[v], S.PREP_HIST, S.TRANSFER_DATASET, S.TRAIN_COMB, self.fx.n2c_run))
        self.assertIn("SHA-256 3dd0c853…", w["model"])
        k, c1 = self.rows_of(doc, "C1 crop check")
        self.assertEqual((k["row"], c1["row"]), ("CROP-cache_0ac7143", "CROP-c1_a80f6f9"))
        for r, e in ((k, "cache_0ac7143"), (c1, "c1_a80f6f9")):
            m = self.fx.crop[e]
            self.assertEqual((r["rejected_bona_fide"], r["accepted_attacks"]), (m["rejected_bona_fide"], m["accepted_attacks"]))
            self.assertEqual((r["apcer"], r["bpcer"], r["acer"], r["auroc"]), tuple(f4(m[kk]) for kk in S.RATES))
            self.assertEqual((r["extractor"], r["preprocessing"], r["run_id"], r["transfer"]), (S.EXTRACTOR_LABELS[e], S.PREP_REF, self.fx.crop_run, S.TRANSFER_SESSION_ALL))
        self.assertEqual(doc["crop_paired"], self.fx.paired)

    def test_annexes_cell_by_cell(self):
        doc = S.build(self.research())
        for r in doc["cross_dataset_rows"]:
            ap, bp, au, eer = self.fx.cross[r["analyzer"]]
            self.assertEqual((r["apcer"], r["bpcer"], r["acer"], r["eer"], r["auroc"]), (f4(ap), f4(bp), f4((ap + bp) / 2), f4(eer), f4(au)))
            arch, rest = r["analyzer"].split("__")
            self.assertEqual(r["model"], S.ARCH_NAMES[arch])
            self.assertEqual(r["source_dataset"], S.CROSS_SOURCES[rest.replace("-pooled", "")])
            expected = ("pooled 80/20 (random over images)" if rest.endswith("-pooled")
                        else "own split by synthetic identity" if rest == "synthaspoof" else "official split of the source dataset")
            self.assertEqual(r["protocol"], expected)
        for r in doc["context_rows"]:
            ap, bp, au, eer = self.fx.context[r["analyzer"]]
            self.assertEqual((r["apcer"], r["bpcer"], r["acer"], r["eer"], r["auroc"]), (f4(ap), f4(bp), f4((ap + bp) / 2), f4(eer), f4(au)))
            self.assertEqual((r["model"], r["trained_on"]), S.CONTEXT_ROWS[r["analyzer"]])
        for r in doc["subject_sensitivity"]:
            d = self.fx.c1[r["analyzer"]]
            self.assertEqual(r["m1_threshold"], f6(d["thr"]))
            self.assertEqual(r["acer_at_0_5"], f6((d["acc"] / 150 + d["rej"] / 150) / 2))
            self.assertEqual(r["acer_at_m1"], f6((d["acc_d"] / 150 + d["rej_d"] / 150) / 2))
            self.assertEqual((r["loso_delta_min"], r["loso_delta_max"]), (f6(d["lo_min"]), f6(d["lo_max"])))
            self.assertEqual(r["sign_changes"], "%d/16" % d["sign_changes"])
            self.assertEqual(r["n_na"], d["n_na"])

    def test_sources_hashed_and_runs_recorded(self):
        research = self.research()
        doc = S.build(research)
        for k, rel in S.SOURCES.items():
            self.assertEqual(doc["sources"][k]["sha256"], S.sha256_file(research / rel))
            self.assertEqual(doc["sources"][k]["path"], rel)
        self.assertEqual(doc["runs"], {"C1": S.C1_RUN, "M1": S.M1_RUN, "N1": self.fx.n1_run, "N1_subjects": "20261024-N1-subjects-a5fcbbc",
                                       "N2_sessions": self.fx.n2s_run, "N2_combined": self.fx.n2c_run, "C1_crop_check": self.fx.crop_run,
                                       "Fold_schema": "20261103-FOLDS-67b7f3d", "Folds_run": self.fx.folds_run,
                                       "Folds_vs_N1": "20261104-FOLDSCOMPARE-abc1234",
                                       "Session_ablation_run": "20261106-SESSABLRUN-d999458", "Patience_12_run": "20261109-SESSABLRUN-6a50265",
                                       "Patience_comparison": "20261109-PATVER-4f2b925"})
        self.assertEqual(set(doc["sources"]), set(S.SOURCES))
        self.assertEqual(len(S.SOURCES), 25)
        for r in doc["rows"]:
            for part in r["source"].split("; "):
                path, prefix = part[:-1].split(" (")
                self.assertEqual(prefix, doc["sources"][next(k for k, v in S.SOURCES.items() if v == path)]["sha256"][:12])

    def test_outputs_deterministic_and_compact_table(self):
        research = self.research()
        a, b = S.outputs(S.build(research)), S.outputs(S.build(research))
        self.assertEqual(a, b)
        self.assertEqual(set(a), {"summary_table.md", "summary_table.csv", "summary_table.json", "summary_table_compact.md",
                                  "cross_dataset_table.md", "context_rows_table.md", "subject_sensitivity_table.md",
                                  "folds_run_table.md", "n1_same_ids_table.md", "patience_table.md", "session_ablation_table.md",
                                  "patience_table_compact.md"})
        compact = [l for l in a["summary_table_compact.md"].splitlines() if l.startswith("| ") and not l.startswith("| block")]
        self.assertEqual(len(compact), 57)
        text = a["summary_table_compact.md"]
        for code in ("T-NUAA", "T-COMB", "S-300", "S-3000", "S-491", "S-03/9", "E-C1", "E-CACHE", "E-WHOLE", "E-CROP", "P-REF", "P-HIST", "same", "session", "session/9", "dataset"):
            self.assertIn("- `%s` (" % code, text)
        d = self.fx.c1["attacknet_v1__nuaa"]
        self.assertIn("| M1 | AttackNet V1 (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | %s (dev EER) | %d / %d | %s | %s | %s | %s | %s | session |"
                      % (f6(d["thr"]), d["rej_d"], d["acc_d"], f4(d["acc_d"] / 150), f4(d["rej_d"] / 150), f4((d["acc_d"] / 150 + d["rej_d"] / 150) / 2), f4(d["auc"]),
                         " | ".join(["\u2014"] * 6) + " | " + S.M1_RUN), text)
        m = self.fx.comb["whole_image"]
        self.assertIn("| N2-C | AttackNet V2.2 combined (coordinator's ckpt) | T-COMB | S-3000 | E-WHOLE | P-HIST | 0.5 (fixed) | %d / %d |" % (m["rejected_bona_fide"], m["accepted_attacks"]), text)
        z = self.fx.n2["43"]["sessions_01_02_491"]
        self.assertIn("| N2-S | LivenessNet, N1 ckpt seed 43 | T-NUAA | S-491 | E-CACHE | P-REF | 0.5 (fixed) | %d / %d |" % (z["rejected_bona_fide"], z["accepted_attacks"]), text)
        self.assertIn("| %s | session/9 |" % self.fx.n2s_run, text)
        self.assertIn("| CROP | AttackNet V2.1 (C1 ckpt 12/09), CPU batch 16 | T-NUAA | S-300 | E-CACHE | P-REF |", text)
        self.assertIn("| N1 | LivenessNet, new CPU training, seed 44 | T-NUAA | S-3000 | E-CACHE | P-REF |", text)
        self.assertIn("- `S-03/9` (test): session 03 test frames of the same 9 training subjects (614 + 924), inside nuaa_cnn_test", text)

    def test_csv_round_trip(self):
        doc = S.build(self.research())
        rows = list(csv.DictReader(io.StringIO(S.outputs(doc)["summary_table.csv"])))
        self.assertEqual(len(rows), 57)
        self.assertEqual(set(rows[0].keys()), set(S.COLUMNS))
        self.assertEqual([r["row"] for r in rows], [r["row"] for r in doc["rows"]])
        self.assertEqual([r["source"] for r in rows], [r["source"] for r in doc["rows"]])


class TestConsistency(Base):
    def test_c1_and_m1_rates_disagree(self):
        fx = self.fx
        research = fx.write(self.root)
        p = research / S.SOURCES["c1_table"]
        d = fx.c1["attacknet_v2_2__nuaa"]
        p.write_text(p.read_text().replace("| %s | 0.6000 |" % f4(d["auc"]), "| %s | 0.6000 |" % f4(d["auc"] + 0.01)), encoding="utf-8")
        self.build_fails(fx, "auc", research)

    def test_c1_and_m1_denominators_disagree(self):
        fx = self.fx
        research = fx.write(self.root)
        p = research / S.SOURCES["c1_table"]
        text = p.read_text().replace("| livenessnet__nuaa | 300 | 150 | 150 |", "| livenessnet__nuaa | 300 | 140 | 150 |")
        p.write_text(text, encoding="utf-8")
        self.build_fails(fx, "denominators differ between the C1 and the M1 table", research)

    def test_m1_counts_not_matching_rates(self):
        fx = self.fx
        research = fx.write(self.root)
        p = research / S.SOURCES["m1_table"]
        d = fx.c1["livenessnet__nuaa"]
        p.write_text(p.read_text().replace("| %d | %d | %s |" % (d["acc"], d["rej"], f6(d["thr"])), "| %d | %d | %s |" % (d["acc"], d["rej"] + 7, f6(d["thr"]))), encoding="utf-8")
        self.build_fails(fx, "BPCER@0.5", research)

    def test_m1_dev_counts_not_matching_rates(self):
        fx = self.fx
        research = fx.write(self.root)
        p = research / S.SOURCES["m1_table"]
        d = fx.c1["attacknet_v2_2__nuaa"]
        p.write_text(p.read_text().replace("| %d | %d | %s |" % (d["acc_d"], d["rej_d"], f4(d["eer"])), "| %d | %d | %s |" % (d["acc_d"] + 5, d["rej_d"], f4(d["eer"]))), encoding="utf-8")
        self.build_fails(fx, "APCER@dev", research)

    def test_m1_denominators_not_the_c1_manifest(self):
        self.fx.mc["c1"] = (300, 140, 160, 16)
        self.build_fails(self.fx, "differ from the c1 manifest")

    def test_subject_counts_differ_from_m1(self):
        self.fx.subj_counts["attacknet_v1__nuaa"]["at_m1"] = (99, 99)
        self.build_fails(self.fx, "counts at at_m1")

    def test_subject_counts_at_0_5_differ_from_m1(self):
        self.fx.subj_counts["livenessnet__nuaa"]["at_0_5"] = (1, 1)
        self.build_fails(self.fx, "counts at at_0_5")

    def test_subject_acer_at_m1_differs(self):
        self.fx.subj_acer_m1["attacknet_v2_1__nuaa"] += 0.01
        self.build_fails(self.fx, "ACER at the M1 threshold")

    def test_subject_threshold_differs_from_m1(self):
        self.fx.subj_thr["attacknet_v1__nuaa"] += 1e-6
        self.build_fails(self.fx, "M1 threshold of attacknet_v1__nuaa differs")

    def test_subject_analyzer_missing(self):
        research = self.edit_json("n1_subjects", lambda d: d["analyzers"].pop("attacknet_v2_2__nuaa"))
        self.build_fails(self.fx, "analyzer attacknet_v2_2__nuaa missing", research)

    def test_subject_table_not_reproduced(self):
        self.fx.subj_reproduced = False
        self.build_fails(self.fx, "M1 table is not reproduced")

    def test_dev_flag_non_informative_is_refused(self):
        self.fx.dev_flag["livenessnet__nuaa"] = "non-informative (dev AUC < 0.5)"
        self.build_fails(self.fx, "dev flag")

    def test_duplicate_rows_are_refused(self):
        self.fx.m1_duplicate_row = "attacknet_v1__nuaa"
        self.build_fails(self.fx, "M1 table: duplicate row attacknet_v1__nuaa")
        fx = Fixture()
        fx.c1_duplicate_row = "depth"
        self.build_fails(fx, "C1 table: duplicate row depth")

    def test_n1_denominators_not_cnn_test(self):
        for s in SEEDS:
            self.fx.n1[s]["test_3000"] = metrics(1400, 1500, 1100, 100, 0.4)
            self.fx.n2[s]["session_03_3000"] = copy.deepcopy(self.fx.n1[s]["test_3000"])
        self.build_fails(self.fx, "N1 seed 42: denominators")

    def test_json_rates_must_match_counts(self):
        self.fx.n1["42"]["test_3000"]["rejected_bona_fide"] -= 3  # tassi invariati
        self.fx.n2["42"]["session_03_3000"]["rejected_bona_fide"] -= 3
        self.build_fails(self.fx, "N1 seed 42 test_3000: bpcer")
        fx = Fixture()
        fx.crop["cache_0ac7143"]["accepted_attacks"] += 1
        self.build_fails(fx, "crop cache_0ac7143: apcer")
        fx = Fixture()
        fx.n2["43"]["session_03_9_subjects"]["acer"] += 0.001
        self.build_fails(fx, "session_03_9_subjects: acer")
        fx = Fixture()
        fx.comb["whole_image"]["rejected_bona_fide"] -= 1
        self.build_fails(fx, "N2 combined whole_image: bpcer")

    def test_auroc_na_only_with_an_absent_class(self):
        self.fx.n2["42"]["session_03_validation_subjects"]["auroc"] = 0.5
        self.build_fails(self.fx, "auroc must be NA exactly when a class is absent")

    def test_n2_subset_sums_and_denominators(self):
        self.fx.n2["42"]["sessions_01_02_fitting_subjects"] = metrics(219, 192, 0, 0, 0.99)
        self.build_fails(self.fx, "n_bona_fide of sessions_01_02_491 is not the sum")
        fx = Fixture()
        m = fx.n2["43"]["session_03_fitting_subjects"]
        fx.n2["43"]["session_03_fitting_subjects"] = metrics(614, 711, m["rejected_bona_fide"], m["accepted_attacks"] + 1, m["auroc"])
        self.build_fails(fx, "accepted_attacks of session_03_9_subjects is not the sum")
        fx = Fixture()
        fx.n2["44"]["session_03_9_subjects"] = metrics(615, 924, 401, 100, 0.36)
        self.build_fails(fx, "denominators of session_03_9_subjects differ between seeds")

    def test_m2_subjects_and_exclusions_cross_checked(self):
        self.fx.m2_subjects["fitted"] = 8
        self.build_fails(self.fx, "cnn_train subjects 7 differ from fitted_subjects 8")
        fx = Fixture()
        fx.m2_exclusions["n_bona_fide"] = 244
        self.build_fails(fx, "differs from the M2 exclusions")
        fx = Fixture()
        fx.target["n_subjects"] = 10
        self.build_fails(fx, "differs from the M2 exclusions")
        fx = Fixture()
        fx.mc["official_train"] = (3491, 1743, 1748, 10)
        self.build_fails(fx, "must equal the official training subjects")

    def test_run_id_must_end_with_the_recorded_commit(self):
        self.fx.crop_run = "20261102-C1CROP-0000000"
        self.build_fails(self.fx, "crop: run id")
        fx = Fixture()
        fx.commits["n2s"] = "deadbeef" * 5
        self.build_fails(fx, "N2 sessions: run id")
        fx = Fixture()
        fx.commits["subj"] = "0000000"
        self.build_fails(fx, "N1 subjects: run id")

    def test_n2_491_denominators_not_the_target(self):
        for s in SEEDS:
            self.fx.n2[s]["sessions_01_02_491"] = metrics(243, 240, 0, 0, 1.0)
            self.fx.n2[s]["sessions_01_02_fitting_subjects"] = metrics(218, 184, 0, 0, 1.0)
        self.build_fails(self.fx, "N2 seed 42 491: denominators")

    def test_n2_session_03_differs_from_n1(self):
        self.fx.n2["43"]["session_03_3000"] = metrics(1500, 1500, 1110, 121, 0.41)
        self.build_fails(self.fx, "session_03_3000 differs from N1")

    def test_n2_weights_differ_from_n1(self):
        self.fx.n2_weights["42"] = "x" * 64
        self.build_fails(self.fx, "weights SHA-256 differs")

    def test_n2_study_not_n1(self):
        self.fx.n2s_n1_study = "20261024-N1-LivenessNet-0000000"
        self.build_fails(self.fx, "n1_study")

    def test_n2_seeds_differ_from_n1(self):
        self.fx.n2_seeds = [42, 43]
        self.build_fails(self.fx, "seeds differ from N1")

    def test_seed_listed_but_missing(self):
        research = self.edit_json("n1_aggregate", lambda d: d["seeds"].pop("44"))
        self.build_fails(self.fx, "N1: seed 44 of train_seeds is missing", research)
        research = self.edit_json("n2_sessions", lambda d: d["seeds"].pop("43"))
        self.build_fails(self.fx, "N2 sessions: seed 43", research)

    def test_combined_reference_differs_from_n1(self):
        research = self.edit_json("n2_combined", lambda d: d["n1_reference"]["seeds"]["44"]["test_3000"].__setitem__("auroc", 0.123))
        self.build_fails(self.fx, "N1 reference seed 44 differs", research)

    def test_combined_variant_missing(self):
        del self.fx.n2_variants["face_crop"]
        self.build_fails(self.fx, "variant face_crop missing")

    def test_combined_denominators_not_cnn_test(self):
        self.fx.comb["whole_image"] = metrics(1500, 1400, 1490, 220, 0.22)
        self.fx.n2_variants = dict(self.fx.comb)
        self.build_fails(self.fx, "N2 combined whole_image: denominators")

    def test_crop_reference_counts_differ_from_c1(self):
        v = self.fx.c1["attacknet_v2_1__nuaa"]
        self.fx.crop_ref = metrics(150, 150, v["rej"] + 1, v["acc"], v["auc"])
        self.build_fails(self.fx, "C1 reference counts differ")

    def test_crop_reference_rates_differ_from_c1(self):
        self.fx.crop_ref["auroc"] += 0.01
        self.build_fails(self.fx, "C1 reference auroc")

    def test_crop_reference_run_is_c1(self):
        self.fx.crop_ref_run = "20260928-C1-seed42-0000000"
        self.build_fails(self.fx, "c1_reference run")

    def test_crop_c1_extractor_row_must_reproduce_reference(self):
        self.fx.crop["c1_a80f6f9"]["auroc"] += 0.01
        self.build_fails(self.fx, "C1 extractor row differs from the C1 reference")
        fx = Fixture()
        v = fx.c1["attacknet_v2_1__nuaa"]
        fx.crop["c1_a80f6f9"] = metrics(150, 150, v["rej"], v["acc"] + 1, v["auc"])
        self.build_fails(fx, "C1 extractor row differs from the C1 reference")

    def test_crop_denominators_differ(self):
        self.fx.crop["cache_0ac7143"] = metrics(140, 150, 120, 5, 0.70)
        self.build_fails(self.fx, "crop cache_0ac7143: denominators")

    def test_crop_extractor_missing(self):
        del self.fx.crop["cache_0ac7143"]
        self.build_fails(self.fx, "extractor cache_0ac7143 missing")

    def test_crop_paired_field_missing(self):
        del self.fx.paired["frames_equal"]
        self.build_fails(self.fx, "paired field frames_equal missing")

    def test_passed_false_and_dirty_true_are_refused(self):
        for key in ("m2", "n1", "n2s", "n2c", "crop"):
            fx = Fixture()
            fx.passed[key] = False
            self.build_fails(fx, "passed is not true")
            fx = Fixture()
            fx.dirty[key] = True
            self.build_fails(fx, "is true")
        fx = Fixture()
        fx.dirty["subj"] = True
        self.build_fails(fx, "dirty is true")

    def test_n1_threshold_not_half(self):
        self.fx.n1_threshold = 0.4
        self.build_fails(self.fx, "expected 0.5")

    def test_cross_rows_count(self):
        del self.fx.cross["attacknet_v1__synthaspoof"]
        self.build_fails(self.fx, "cross-dataset CNN rows found, expected 16")

    def test_context_row_missing(self):
        del self.fx.context["fourier"]
        self.build_fails(self.fx, "context row fourier missing")

    def test_c1_row_missing(self):
        self.fx.c1_missing_row = "livenessnet__nuaa"
        self.build_fails(self.fx, "row livenessnet__nuaa missing")

    def test_m1_row_missing(self):
        self.fx.m1_missing_row = "attacknet_v2_1__nuaa"
        self.build_fails(self.fx, "M1 table: row attacknet_v2_1__nuaa missing")

    def test_missing_columns(self):
        self.fx.m1_missing_column = "dev flag"
        self.build_fails(self.fx, "M1 table: missing columns")
        fx = Fixture()
        fx.c1_missing_column = "eer"
        self.build_fails(fx, "C1 table: missing columns")

    def test_source_missing(self):
        research = self.research()
        (research / S.SOURCES["crop"]).unlink()
        self.build_fails(self.fx, "source crop missing", research)

    def test_required_json_fields(self):
        research = self.edit_json("n2_combined", lambda d: d.pop("n1_reference"))
        self.build_fails(self.fx, "field 'n1_reference' missing", research)
        research = self.edit_json("n1_subjects", lambda d: d.pop("run_id"))
        self.build_fails(self.fx, "field 'run_id' missing", research)
        research = self.edit_json("m2_manifests", lambda d: d.pop("exclusions"))
        self.build_fails(self.fx, "field 'exclusions' missing", research)
        research = self.edit_json("m2_manifests", lambda d: d["manifests"].pop("m1_dev"))
        self.build_fails(self.fx, "M2 manifests: m1_dev missing", research)

    def test_metric_field_missing(self):
        del self.fx.comb["face_crop"]["auroc"]
        self.build_fails(self.fx, "metric fields missing")

    def test_count_not_integer(self):
        fx = self.fx
        research = fx.write(self.root)
        p = research / S.SOURCES["m1_table"]
        d = fx.c1["livenessnet__nuaa"]
        p.write_text(p.read_text().replace("| %d | %d | %s |" % (d["acc"], d["rej"], f6(d["thr"])), "| %d | abc | %s |" % (d["acc"], f6(d["thr"]))), encoding="utf-8")
        self.build_fails(fx, "is not an integer count", research)


class TestFolds(Base):
    """Run a fold di soggetti: righe, annesso, tabella compatta e ogni controllo di coerenza su una sorgente guasta."""

    def all_of(self, k, s):
        return self.fx.fagg["per_fold_seed"][k][s]["all"]

    def test_fold_rows_cell_by_cell(self):
        doc = S.build(self.research())
        rows = [r for r in self.rows_of(doc, S.FOLDS_BLOCK) if not r["row"].startswith("FOLDS-pooled")]
        self.assertEqual([r["row"] for r in rows], ["FOLDS-f%s-seed%s" % (k, s) for k in "1234" for s in SEEDS])
        for r in rows:
            k, s = r["row"][len("FOLDS-f"):].split("-seed")
            m, tr = self.all_of(k, s), self.fx.fagg["training"]["fold%s_seed%s" % (k, s)]
            self.assertEqual((r["n_bona_fide"], r["n_attack"], r["rejected_bona_fide"], r["accepted_attacks"],
                              r["rejected_bona_fide_dev"], r["accepted_attacks_dev"]),
                             (m["n_bona_fide"], m["n_attack"], m["rejected_bona_fide_05"], m["accepted_attacks_05"],
                              m["rejected_bona_fide_dev"], m["accepted_attacks_dev"]))
            self.assertEqual((r["apcer"], r["bpcer"], r["acer"], r["auroc"], r["apcer_dev"], r["bpcer_dev"], r["acer_dev"]),
                             tuple(f4(m[x]) for x in ("apcer_05", "bpcer_05", "acer_05", "auroc", "apcer_dev", "bpcer_dev", "acer_dev")))
            self.assertEqual((r["threshold"], r["threshold_dev"], r["bootstrap"]), ("0.5", f6(tr["dev_threshold"]), ""))
            self.assertEqual((r["run_id"], r["transfer"], r["extractor"], r["preprocessing"]),
                             (self.fx.folds_run, S.TRANSFER_FOLDS, S.EXTRACTOR_FOLDS, S.PREP_REF))
            pf = self.fx.fsch["per_fold"][k]
            self.assertEqual(r["model"], "LivenessNet, new CPU training, held-out fold %s, seed %s (fitting folds %s, dev fold %s; "
                                         "restored epoch %d of 12, early stopping; min val_loss %.6f)"
                             % (k, s, " and ".join(str(x) for x in pf["fitting_folds"]), pf["dev_fold"], tr["restored_epoch"], tr["best_val_loss"]))
            n_bf, n_att = sum(c[0] for c in self.fx.fold_cells[k].values()), sum(c[1] for c in self.fx.fold_cells[k].values())
            self.assertEqual(r["test_set"], "held-out fold %s: all %s frames (%s + %s) of its 4 subjects, sessions 01-03"
                             % (k, "{:,}".format(n_bf + n_att), "{:,}".format(n_bf), "{:,}".format(n_att)))
            self.assertEqual([p.split(" (")[0] for p in r["source"].split("; ")],
                             [S.SOURCES["folds_aggregate"], S.SOURCES["folds_record"], S.SOURCES["folds_schema"]])
        self.assertEqual(doc["texts"]["folds_train"], "NUAA, all three sessions (01-03): 2,100 fitting frames of the 8 subjects of the two "
                                                      "fitting folds, 350 per class x session cell (deterministic spread rule)")

    def test_pooled_rows_with_bootstrap(self):
        doc = S.build(self.research())
        rows = [r for r in self.rows_of(doc, S.FOLDS_BLOCK) if r["row"].startswith("FOLDS-pooled")]
        self.assertEqual([r["row"] for r in rows], ["FOLDS-pooled-seed" + s for s in SEEDS])
        for r, s in zip(rows, SEEDS):
            m, b = self.fx.fagg["pooled_per_seed"][s]["all"], self.fx.fagg["bootstrap_per_seed"][s]["all"]
            self.assertEqual((r["n_bona_fide"], r["n_attack"]), (5105, 7509))
            self.assertEqual((r["rejected_bona_fide"], r["accepted_attacks"], r["rejected_bona_fide_dev"], r["accepted_attacks_dev"]),
                             (m["rejected_bona_fide_05"], m["accepted_attacks_05"], m["rejected_bona_fide_dev"], m["accepted_attacks_dev"]))
            self.assertEqual((r["acer"], r["acer_dev"], r["auroc"]), (f4(m["acer_05"]), f4(m["acer_dev"]), f4(m["auroc"])))
            self.assertEqual(r["bootstrap"], "ACER 0.5 %s-%s; ACER dev %s-%s; AUROC %s-%s"
                             % (f4(b["acer_05"]["low"]), f4(b["acer_05"]["high"]), f4(b["acer_dev"]["low"]), f4(b["acer_dev"]["high"]),
                                f4(b["auroc"]["low"]), f4(b["auroc"]["high"])))
            self.assertEqual(r["threshold_dev"], S.THR_DEV_POOLED)
            self.assertEqual(r["test_set"], doc["texts"]["folds_test_all"])
        self.assertEqual(doc["texts"]["folds_test_all"], "the 4 held-out folds pooled: all 12,614 official NUAA frames (5,105 + 7,509), "
                                                         "16 subjects, sessions 01-03, each frame scored by the model of its fold")
        self.assertEqual(doc["folds_run"]["test_frames_per_fold"], self.fx.fold_totals)
        self.assertEqual((doc["folds_run"]["train_seconds_min"], doc["folds_run"]["train_seconds_max"]), (710.0, 742.0))

    def test_annex_every_stratum_and_summary(self):
        text = S.outputs(S.build(self.research()))["folds_run_table.md"]
        lines = [l for l in text.splitlines() if l.startswith("| ") and not l.startswith("| fold |")
                 and not l.startswith("| seed |") and not l.startswith("| stratum |") and not l.startswith("| model |")]
        self.assertEqual(len(lines), 48 + 16 + 12 + 4 + 12)
        m = self.fx.fagg["per_fold_seed"]["4"]["43"]["02"]
        self.assertIn("| 4 | 43 | 02 | %d | %d | %d | %d / %d | %s | %s | %s | %d / %d | %s | %s | %s | %s |"
                      % (m["n_bona_fide"], m["n_attack"], m["n_subjects"], m["rejected_bona_fide_05"], m["accepted_attacks_05"],
                         f4(m["bpcer_05"]), f4(m["apcer_05"]), f4(m["acer_05"]), m["rejected_bona_fide_dev"], m["accepted_attacks_dev"],
                         f4(m["bpcer_dev"]), f4(m["apcer_dev"]), f4(m["acer_dev"]), f4(m["auroc"])), text)
        e = self.fx.fagg["per_fold_summary_over_seeds"]["2"]["03"]
        self.assertIn("| 2 | 03 | %s |" % " | ".join("%s (%s-%s)" % (f4(e[k]["median"]), f4(e[k]["min"]), f4(e[k]["max"]))
                                                      for k in ("acer_05", "acer_dev", "auroc")), text)
        e = self.fx.fagg["pooled_summary_over_seeds"]["all"]
        self.assertIn("| all | %s (%s-%s) |" % (f4(e["acer_05"]["median"]), f4(e["acer_05"]["min"]), f4(e["acer_05"]["max"])), text)
        self.assertIn("| fold3_seed44 | 2100 | 600 | 12 |", text)

    def test_compact_rows_and_legend(self):
        text = S.outputs(S.build(self.research()))["summary_table_compact.md"]
        m, tr = self.all_of("1", "43"), self.fx.fagg["training"]["fold1_seed43"]
        self.assertIn("| FOLDS | LivenessNet, fold 1, seed 43 | T-FOLD | S-FOLD-1 | E-CACHE | P-REF | 0.5 (fixed) | %d / %d | %s | %s | %s | %s | "
                      "%s (EER, fold dev) | %d / %d | %s | %s | %s | — | %s | subject-held-out (sessions represented) |"
                      % (m["rejected_bona_fide_05"], m["accepted_attacks_05"], f4(m["apcer_05"]), f4(m["bpcer_05"]), f4(m["acer_05"]),
                         f4(m["auroc"]), f6(tr["dev_threshold"]), m["rejected_bona_fide_dev"], m["accepted_attacks_dev"],
                         f4(m["apcer_dev"]), f4(m["bpcer_dev"]), f4(m["acer_dev"]), self.fx.folds_run), text)
        b = self.fx.fagg["bootstrap_per_seed"]["44"]["all"]
        self.assertIn("| FOLDS | LivenessNet, 4 fold models, seed 44 (pooled) | T-FOLD | S-FOLD-ALL |", text)
        self.assertIn("| 4 fold values (EER, fold dev) |", text)
        self.assertIn("| %s-%s; %s-%s; %s-%s | %s |" % (f4(b["acer_05"]["low"]), f4(b["acer_05"]["high"]), f4(b["acer_dev"]["low"]),
                                                       f4(b["acer_dev"]["high"]), f4(b["auroc"]["low"]), f4(b["auroc"]["high"]),
                                                       self.fx.folds_run), text)
        for code in ("T-FOLD", "S-FOLD-1", "S-FOLD-2", "S-FOLD-3", "S-FOLD-4", "S-FOLD-ALL", "(EER, fold dev)", "(dev EER)", "0.5 (fixed)",
                     "subject-held-out (sessions represented)"):
            self.assertIn("- `%s` (" % code, text)
        self.assertIn("- `T-FOLD` (training): fitting 2,100 + dev 600 frames of 8 + 4 subjects outside the held-out fold", text)
        self.assertIn("- `subject-held-out, N1 test frames` (transfer):", text)

    def test_denominators_must_equal_the_fold_schema(self):
        c = self.fx.fsch["per_fold"]["2"]["roles"]["test"]["counts"]
        c["03/attack"]["frames"] += 1
        c["total"]["frames"] += 1
        self.build_fails(self.fx, "folds fold 2 seed 42 stratum 03: denominators 1020/1360 differ from the fold schema 1020/1361")

    def test_schema_total_and_official_lists(self):
        self.fx.fsch["per_fold"]["1"]["roles"]["test"]["counts"]["total"]["frames"] += 5
        self.build_fails(self.fx, "fold 1 test total")
        fx = Fixture()
        fx.mc["official_test"] = (9124, 3363, 5761, 16)
        self.build_fails(fx, "do not add up to the official lists of M2")
        fx = Fixture()
        fx.mc["official_test"] = (9123, 3362, 5761, 17)
        self.build_fails(fx, "16 held-out subjects, the official test lists have 17")

    def test_stratum_all_must_be_the_sum_of_the_sessions(self):
        m = self.all_of("3", "44")
        m["rejected_bona_fide_dev"] += 1
        refresh(m)
        self.build_fails(self.fx, "folds fold 3 seed 44: stratum all at dev is not the sum of the session strata")

    def test_pooled_must_be_the_sum_of_the_folds(self):
        m = self.fx.fagg["pooled_per_seed"]["43"]["03"]
        m["accepted_attacks_05"] += 1
        refresh(m)
        self.build_fails(self.fx, "folds pooled seed 43 stratum 03 at 05: counts are not the sum of the 4 folds")

    def test_rates_must_match_counts(self):
        self.fx.fagg["per_fold_seed"]["1"]["42"]["01"]["accepted_attacks_dev"] += 1  # tassi invariati
        self.build_fails(self.fx, "folds fold 1 seed 42 stratum 01 at dev: apcer")
        fx = Fixture()
        fx.fagg["pooled_per_seed"]["44"]["all"]["auroc"] = None
        self.build_fails(fx, "auroc must be NA exactly when a class is absent")

    def test_summaries_must_equal_the_per_seed_values(self):
        self.fx.fagg["per_fold_summary_over_seeds"]["2"]["03"]["auroc"]["values"][1] += 0.001
        self.build_fails(self.fx, "folds summary fold 2 stratum 03 auroc: values")
        fx = Fixture()
        fx.fagg["pooled_summary_over_seeds"]["all"]["acer_dev"]["median"] += 0.001
        self.build_fails(fx, "folds pooled summary stratum all acer_dev: median or range")
        fx = Fixture()
        del fx.fagg["pooled_summary_over_seeds"]["01"]["bpcer_05"]
        self.build_fails(fx, "folds pooled summary stratum 01: bpcer_05 missing")

    def test_bootstrap_parameters_and_intervals(self):
        self.fx.fagg["bootstrap_per_seed"]["42"]["all"]["resamples"] = 999
        self.build_fails(self.fx, "folds bootstrap seed 42 stratum all: subjects or parameters")
        fx = Fixture()
        b = fx.fagg["bootstrap_per_seed"]["43"]["03"]["auroc"]
        b["low"], b["high"] = b["high"], b["low"] - 0.1
        self.build_fails(fx, "folds bootstrap seed 43 stratum 03: auroc interval missing or reversed")
        fx = Fixture()
        del fx.fagg["bootstrap_per_seed"]["44"]["02"]
        self.build_fails(fx, "folds bootstrap seed 44 stratum 02 missing")

    def test_record_must_describe_the_same_run(self):
        for edit, needle in ((lambda r: r.__setitem__("config_sha256", "8" * 64), "config_sha256"),
                             (lambda r: r["schema_run"].__setitem__("run_id", "20261103-FOLDS-0000000"), "schema run"),
                             (lambda r: r["checks"]["schema_config"].__setitem__("sha256", "e" * 64), "checked schema configuration"),
                             (lambda r: r["checks"]["schema"]["test_per_fold"].reverse(), "test rows per fold"),
                             (lambda r: r["extraction"].__setitem__("n", 6613), "6613 extracted + 6000 cache frames"),
                             (lambda r: r["cache_check"]["test"].__setitem__("different", 1), "not reproduced bit for bit"),
                             (lambda r: r["checks"]["originals"].__setitem__("located_and_hashed", 12000), "12000 originals verified"),
                             (lambda r: r.__setitem__("wall_seconds", 21600.5), "above the cap"),
                             (lambda r: r.__setitem__("models", 11), "folds and 11 models"),
                             (lambda r: r.__setitem__("train_seeds", [42, 43, 45]), "seeds"),
                             (lambda r: r["timings"].pop("train fold2_seed43"), "no timing for train fold2_seed43")):
            fx = Fixture()
            edit(fx.frec)
            self.build_fails(fx, needle)

    def test_clean_tree_and_delivered_configuration(self):
        for target, key, value, needle in (("fagg", "dirty", True, "folds aggregate: dirty is not false"),
                                           ("frec", "dirty", None, "folds run record: dirty is not false"),
                                           ("fsch", "dirty", True, "fold schema: dirty is not false"),
                                           ("fagg", "delivered_configuration", False, "not a run of the delivered configuration"),
                                           ("frec", "command", "check-inputs", "not a run of the delivered configuration"),
                                           ("fagg", "commit", "0000000", "folds aggregate: run id")):
            fx = Fixture()
            getattr(fx, target)[key] = value
            self.build_fails(fx, needle)

    def test_training_cards(self):
        self.fx.fagg["training"]["fold4_seed42"]["n_fitting"] = 2000
        self.build_fails(self.fx, "folds fold4_seed42: 2000 fitting / 600 dev frames, the fold schema has 2100 / 600")
        fx = Fixture()
        fx.fagg["training"]["fold1_seed44"]["restored_epoch"] = 13
        self.build_fails(fx, "restored epoch 13 outside 1..12")
        fx = Fixture()
        del fx.fagg["training"]["fold3_seed43"]["dev_threshold"]
        self.build_fails(fx, "training field dev_threshold of fold3_seed43 missing")
        fx = Fixture()
        del fx.fagg["training"]["fold2_seed44"]
        self.build_fails(fx, "training of fold2_seed44 missing")

    def test_structure_and_fold_sizes(self):
        self.fx.fagg["strata"] = ["01", "02", "all"]
        self.build_fails(self.fx, "folds aggregate: strata")
        fx = Fixture()
        del fx.fagg["per_fold_seed"]["4"]["44"]
        self.build_fails(fx, "fold 4 seed 44 missing")
        fx = Fixture()
        fx.fsch["per_fold"]["3"]["roles"]["dev"]["n_subjects"] = 5
        self.build_fails(fx, "fold schema: dev subjects differ between folds")
        fx = Fixture()
        fx.fsch["train_seeds"] = [42, 43]
        self.build_fails(fx, "seeds")
        fx = Fixture()
        fx.fagg["per_fold_seed"]["2"]["43"]["all"]["n_subjects"] = 3
        self.build_fails(fx, "folds fold 2 seed 43: 3 subjects, the fold schema has 4")
        research = self.edit_json("folds_record", lambda d: d.pop("cache_check"))
        self.build_fails(self.fx, "field 'cache_check' missing", research)


class TestCompare(Base):
    """Confronto con N1 sugli stessi 3.000 frame: righe, annesso, run ufficiale richiesto, ogni controllo su una sorgente guasta."""

    def guasto(self, edit, needle):
        fx = Fixture()
        fx.compare = fx.compare_json()
        edit(fx.compare)
        self.build_fails(fx, needle)

    def test_rows_cell_by_cell_and_no_duplicated_n1(self):
        research = self.research()
        doc = S.build(research)
        cmp = self.fx.compare_json()
        rows = self.rows_of(doc, S.COMPARE_BLOCK)
        self.assertEqual([r["row"] for r in rows], ["FOLDSN1-seed" + s for s in SEEDS])
        for r, s in zip(rows, SEEDS):
            m, b = cmp["per_seed"][s]["folds"], cmp["bootstrap_per_seed"][s]
            self.assertEqual((r["n_bona_fide"], r["n_attack"], r["rejected_bona_fide"], r["accepted_attacks"],
                              r["rejected_bona_fide_dev"], r["accepted_attacks_dev"]),
                             (1500, 1500, m["rejected_bona_fide_05"], m["accepted_attacks_05"], m["rejected_bona_fide_dev"], m["accepted_attacks_dev"]))
            self.assertEqual((r["apcer"], r["bpcer"], r["acer"], r["auroc"], r["apcer_dev"], r["bpcer_dev"], r["acer_dev"]),
                             tuple(f4(m[x]) for x in ("apcer_05", "bpcer_05", "acer_05", "auroc", "apcer_dev", "bpcer_dev", "acer_dev")))
            self.assertEqual(r["bootstrap"], "ACER 0.5 %s-%s; ACER dev %s-%s; AUROC %s-%s"
                             % tuple(f4(b[k][e]) for k in ("folds_acer_05", "folds_acer_dev", "folds_auroc") for e in ("low", "high")))
            self.assertEqual((r["test_set"], r["transfer"], r["run_id"], r["threshold_dev"]),
                             (doc["texts"]["test_3000"], S.TRANSFER_FOLDS_N1, "20261104-FOLDSCOMPARE-abc1234", S.THR_DEV_POOLED))
            self.assertEqual(r["source"].split(" (")[0], S.SOURCES["folds_n1_compare"])
        self.assertEqual(len(self.rows_of(doc, "N1")), 3)  # le righe N1 non sono duplicate
        self.assertEqual(doc["sources"]["folds_n1_compare"]["sha256"], S.sha256_file(research / S.SOURCES["folds_n1_compare"]))
        text = S.outputs(doc)["summary_table_compact.md"]
        self.assertIn("| FOLDS-N1 | LivenessNet, 4 fold models, seed 43 (N1 test frames) | T-FOLD | S-3000 | E-CACHE |", text)

    def test_annex_copied_from_the_aggregate(self):
        doc = S.build(self.research())
        text = S.outputs(doc)["n1_same_ids_table.md"]
        cmp = self.fx.compare_json()
        p, b = cmp["per_seed"]["44"], cmp["bootstrap_per_seed"]["44"]
        n = p["n1"]
        self.assertIn("| 44 | N1, 0.5 | %d / %d | %s | %s | %s [%s, %s] | %s [%s, %s] |"
                      % (n["rejected_bona_fide_05"], n["accepted_attacks_05"], f4(n["bpcer_05"]), f4(n["apcer_05"]), f4(n["acer_05"]),
                         f4(b["n1_acer_05"]["low"]), f4(b["n1_acer_05"]["high"]), f4(n["auroc"]), f4(b["n1_auroc"]["low"]),
                         f4(b["n1_auroc"]["high"])), text)
        self.assertIn("| 44 | folds, dev |", text)
        d = p["difference_folds_minus_n1"]
        self.assertIn("| 44 | %s [%s, %s] |" % (f4(d["diff_acer_05"]), f4(b["diff_acer_05"]["low"]), f4(b["diff_acer_05"]["high"])), text)
        c = p["paired_decisions"]["dev"]
        self.assertIn("| 44 | dev | %s | %s | %d |" % tuple([" / ".join(str(c[cls][k]) for k in ("both_correct", "n1_only_correct",
                                                                                     "folds_only_correct", "both_wrong"))
                                                            for cls in ("bona_fide", "attack")] + [c["decisions_changed"]]), text)
        self.assertIn("| 43 | 0.5 | error rate | 12 | 0 | 4 | 0 |", text)
        v = cmp["per_seed"]["42"]["folds"]["acer_dev"]
        self.assertIn("| acer_dev | 42 | %.2f %% | %.5f |" % (round(100 * v, 2), v), text)
        a = cmp["per_seed"]["43"]["folds"]["auroc"]
        self.assertIn("| auroc | 43 | %.5f | %.5f |" % (round(a, 5), a), text)
        self.assertIn("Comparison: agreement at the digits given by the coordinator.", text)
        self.assertIn("recipe: generator restarted for every seed; same resamples for both experiments.", text)
        self.assertIn("N1 AUROC is below 0.5 on these frames (0.4000-0.4200, as published by N1); the AUROC difference includes that.", text)
        lines = [l for l in text.splitlines() if l.startswith("| ") and not l.startswith(("| seed |", "| measure |"))]
        self.assertEqual(len(lines), 9 + 3 + 8 + 6 + 12 + 9)

    def test_trial_run_is_refused(self):
        self.guasto(lambda d: d.__setitem__("dirty", True), "folds vs N1: dirty is True")
        self.guasto(lambda d: d.__setitem__("run_id", "20261104-FOLDSCOMPARE-abc1234-dev"), "ends with -dev (trial run)")
        self.fx.compare = self.fx.compare_json()
        self.fx.compare.update(dirty=True, run_id="20261104-FOLDSCOMPARE-abc1234-dev")
        research = self.research()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = S.main(["build", "--research", str(research), "--out", str(self.root / "t")])
        self.assertEqual(code, 1)
        self.assertIn("the official run must replace it", out.getvalue())
        self.assertFalse((self.root / "t").exists())

    def test_n1_auroc_sentence_only_when_below_half(self):
        fx = Fixture()
        for i, s in enumerate(SEEDS):
            m = fx.n1[s]["test_3000"]
            m["auroc"] = 0.45 + i / 10
            fx.n2[s]["session_03_3000"]["auroc"] = m["auroc"]
        text = S.outputs(S.build(fx.write(self.root)))["n1_same_ids_table.md"]
        self.assertNotIn("N1 AUROC is below 0.5", text)

    def test_inputs_and_run_are_cross_checked(self):
        for edit, needle in ((lambda d: d["inputs"]["manifest"].__setitem__("sha256", "1" * 64), "manifest inputs"),
                             (lambda d: d["inputs"]["manifest"].__setitem__("schema_config_sha256", "e" * 64), "manifest inputs"),
                             (lambda d: d["inputs"]["folds_run"].__setitem__("run_id", "20261103-FOLDSRUN-0000000"), "folds run"),
                             (lambda d: d["inputs"]["n1_run"].__setitem__("run_id", "20261024-N1-LivenessNet-0000000"), "N1 run"),
                             (lambda d: d["inputs"]["folds_run"]["rows_per_seed"].__setitem__("43", 12613), "seed 43: score rows"),
                             (lambda d: d["inputs"]["folds_run"]["test_subjects_per_fold"].__setitem__("2", 5), "test subjects per fold"),
                             (lambda d: d["declarations"].__setitem__("post_hoc", False), "declaration post_hoc"),
                             (lambda d: d.__setitem__("delivered_configuration", False), "not a run of the delivered configuration"),
                             (lambda d: d.__setitem__("commit", "fff0000"), "folds vs N1: run id"),
                             (lambda d: d.pop("control_values"), "field 'control_values' missing")):
            self.guasto(edit, needle)

    def test_n1_values_must_equal_the_n1_rows(self):
        self.guasto(lambda d: d["per_seed"]["42"]["n1"].__setitem__("auroc", 0.123), "seed 42: the N1 metrics differ from the N1 row")

        def counts(d):
            n = d["per_seed"]["43"]["n1"]
            n["rejected_bona_fide_05"] += 1
            n["bpcer_05"] = n["rejected_bona_fide_05"] / 1500
            n["acer_05"] = (n["bpcer_05"] + n["apcer_05"]) / 2
        self.guasto(counts, "seed 43: the N1 metrics differ from the N1 row")

    def test_folds_counts_inside_the_pooled_session_03(self):
        def big(d):
            f = d["per_seed"]["44"]["folds"]
            f["rejected_bona_fide_05"] = 500
            refresh(f)
        self.guasto(big, "folds vs N1 seed 44 at 05: counts exceed the pooled session 03 of the folds run")
        self.guasto(lambda d: d["per_seed"]["44"].__setitem__("folds", fstratum(1500, 1400, 1, 1, 1, 1, 0.9, 16)),
                    "folds vs N1 seed 44 folds: denominators")

    def test_differences_paired_decisions_and_subjects(self):
        self.guasto(lambda d: d["per_seed"]["42"]["difference_folds_minus_n1"].__setitem__("diff_auroc", 0.5), "diff_auroc 0.5 is not folds minus N1")
        self.guasto(lambda d: d["per_seed"]["43"]["paired_decisions"]["05"]["bona_fide"].__setitem__("both_wrong", 0),
                    "seed 43 at 05: paired decisions of bona_fide do not match the counts")
        self.guasto(lambda d: d["per_seed"]["44"]["paired_decisions"]["dev"].__setitem__("decisions_changed", 1), "seed 44 at dev: decisions changed 1")
        self.guasto(lambda d: d["per_seed"]["42"]["per_subject_counts"]["comparison"]["dev"]["acer"].__setitem__("not_defined", 7),
                    "seed 42 at dev: subject counts of acer do not add up")
        self.guasto(lambda d: d["per_seed"]["42"]["per_subject_counts"].__setitem__("n_subjects", 15), "15 subjects")

    def test_summary_bootstrap_and_control_values(self):
        self.guasto(lambda d: d["summary_over_seeds"]["n1_auroc"]["values"].reverse(), "summary n1_auroc: missing or values differ")
        self.guasto(lambda d: d["summary_over_seeds"]["diff_acer_dev"].__setitem__("median", 0.0), "summary diff_acer_dev: median or range")
        self.guasto(lambda d: d["bootstrap_per_seed"]["43"].__setitem__("seed", 1), "bootstrap seed 43: missing, or subjects or parameters")
        self.guasto(lambda d: d["bootstrap_per_seed"]["44"]["diff_auroc"].__setitem__("low", 2.0), "seed 44: diff_auroc interval missing or reversed")
        self.guasto(lambda d: d["control_values"]["control"]["auroc"].__setitem__(1, 0.5), "control values of auroc not reproduced at the digits")
        self.guasto(lambda d: d["control_values"]["control"]["acer_dev_percent"].__setitem__(0, round(d["control_values"]["control"]["acer_dev_percent"][0] + 0.01, 2)),
                    "control values of acer_dev not reproduced at the digits given by the coordinator")
        self.guasto(lambda d: d["control_values"].pop("comparison"), "comparison rule missing")
        self.guasto(lambda d: d["bootstrap"].pop("recipe"), "or the recipe is not declared")
        self.guasto(lambda d: d["control_values"]["computed"]["acer_05"].__setitem__(0, 0.3), "computed control values of acer_05 differ")
        self.guasto(lambda d: d["control_values"].__setitem__("reproduced", False), "not declared reproduced")

    def test_source_is_required(self):
        research = self.research()
        (research / S.SOURCES["folds_n1_compare"]).unlink()
        self.build_fails(self.fx, "source folds_n1_compare missing", research)


LR_TEXT = {0.001: "0.001", 0.0005: "0.0005", 0.00025: "0.00025", 0.000125: "0.000125", 6.25e-05: "0.0000625"}


def fna(v):
    return "NA" if v is None else "%.4f" % v


def expected_lines(fx):
    """Le righe di riepilogo attese, ricostruite dalla fixture con le soglie di CONCLUSION.md (epoca 10, accuratezza 0,9)."""
    sd = [fx.sides[m] for m in fx.abl_models]
    ten = [i for i in range(36) if i in fx.ten]
    oth = [i for i in range(36) if i not in fx.ten]
    ch = [i for i, m in enumerate(fx.abl_models) if fx.changed(m)]
    n_ten, n_oth = len(ten), len(oth)

    def c(idx, f):
        return sum(1 for i in idx if f(i))

    def med(v, form="%g"):
        return "%s (%s-%s)" % (form % statistics.median(v), form % min(v), form % max(v))
    A = range(36)
    ep = [[sd[i][r]["epochs_run"] for i in A] for r in (0, 1)]

    def span(v):
        return ("%d" % min(v)) if min(v) == max(v) else "%d-%d" % (min(v), max(v))
    l1 = ["Models: 36 (3 arms x 4 folds x 3 seeds); complete: 36 with patience 4, 36 with patience 12.",
          "In the ten original degenerate trainings: %d." % n_ten]
    for r, p in ((0, 4), (1, 12)):
        l1.append("Degenerate with patience %d: %d of 36 (%d of the ten)." % (p, c(A, lambda i: sd[i][r]["degenerate"]), c(ten, lambda i: sd[i][r]["degenerate"])))
    l1.append("Shared histories identical: 36 of 36 (shared epochs = epochs run with patience 4 in 36 of 36).")
    for r, p in ((0, 4), (1, 12)):
        l1.append("Epochs run with patience %d: %s, %d of 36 models at 12." % (p, span(ep[r]), ep[r].count(12)))
    for r, p in ((0, 4), (1, 12)):
        l1.append("With a recovery (an epoch meeting the recovery criterion) with patience %d: %d of 36 (%d of the ten)."
                  % (p, c(A, lambda i: sd[i][r]["recovered"]), c(ten, lambda i: sd[i][r]["recovered"])))
    for r, p in ((0, 4), (1, 12)):
        l1.append("Recovery holds at the last epoch with patience %d: %d of 36 (%d of the ten)."
                  % (p, c(A, lambda i: sd[i][r]["holding_last"]), c(ten, lambda i: sd[i][r]["holding_last"])))
    l1 += ["First recovery of the ten with patience 12: median (min-max) %s." % med([sd[i][1]["first_recovery"] for i in ten]),
           "First recovery of the ten at epoch 10 or later with patience 12: %d of the ten (cut-off of CONCLUSION.md)."
           % c(ten, lambda i: sd[i][1]["first_recovery"] >= 10),
           "Restored epoch changed: %d of 36 (%d of the ten, %d of the other %d)."
           % (len(ch), c(ch, lambda i: i in fx.ten), c(ch, lambda i: i not in fx.ten), n_oth),
           "Stopped before epoch 12 with patience 4: %d models, %d outside the ten, of which %d changed restored epoch."
           % (c(A, lambda i: ep[0][i] < 12), c(oth, lambda i: ep[0][i] < 12), c(ch, lambda i: i not in fx.ten and ep[0][i] < 12))]
    for group, idx in (("the ten", ten), ("the other %d" % n_oth, oth)):
        for r, p in ((0, 4), (1, 12)):
            l1.append("Dev accuracy at the checkpoint, %s, median (min-max) with patience %d: %s."
                      % (group, p, med([sd[i][r]["dev_acc_at_checkpoint"] for i in idx], "%.4f")))
    for group, idx in (("the ten", ten), ("the other %d" % n_oth, oth)):
        l1.append("Dev accuracy at the checkpoint below 0.9 with patience 12, %s: %d (cut-off of CONCLUSION.md)."
                  % (group, c(idx, lambda i: sd[i][1]["dev_acc_at_checkpoint"] < 0.9)))
    l2 = ["Same test values and dev threshold in both runs (checked): all %d models whose restored epoch did not change." % (36 - len(ch)),
          "The differences below come from the %d models whose restored epoch changed (\"changed\" below)." % len(ch)]
    P0, P1 = fx.abl[0]["agg"]["per_arm_fold_seed"], fx.abl[1]["agg"]["per_arm_fold_seed"]
    for st in ("03", "all"):
        for key, label in (("acer_05", "ACER at 0.5"), ("acer_dev", "ACER at the dev threshold"), ("auroc", "AUROC")):
            def rel(i):
                a, k, s = fx.abl_models[i]
                x, y = P0[a][k][s][st][key], P1[a][k][s][st][key]
                if x is None or y is None:
                    return "NA"
                return "lower" if y < x else "higher" if y > x else "equal"
            for w in ("lower", "equal", "higher"):
                l2.append("Stratum %s, %s, patience 12 vs 4: %s in %d of 36 (%d of the %d changed, %d of the ten)."
                          % (st, label, w, c(A, lambda i: rel(i) == w), c(ch, lambda i: rel(i) == w), len(ch), c(ten, lambda i: rel(i) == w)))
            na = c(A, lambda i: rel(i) == "NA")
            if na:
                l2.append("Stratum %s, %s, patience 12 vs 4: NA in %d of 36 (a class absent)." % (st, label, na))
    vs = [{(cn, st, m): fx.abl[r]["agg"]["comparisons"][cn]["verdicts"][st][m] for cn, _, _, _ in COMPS for st in STRATA4 for m in KEYS7}
          for r in (0, 1)]
    cons = [sum(v["verdict"] == "consistent" for v in d.values()) for d in vs]
    diff = [k for k in vs[0] if (vs[0][k]["verdict"], vs[0][k]["direction"]) != (vs[1][k]["verdict"], vs[1][k]["direction"])]

    def counts(n):
        return ", ".join("%s %d" % (w, x) for w, x in sorted({"consistent": n, "inconclusive": 84 - n}.items()) if x)
    l3 = ["Verdicts: 84 per run (3 comparisons x 4 strata x 7 metrics); patience 4: %s; patience 12: %s." % (counts(cons[0]), counts(cons[1]))]
    for r, p in ((0, 4), (1, 12)):
        l3.append("Primary verdicts (A* - B, stratum 03: auroc, acer_dev) with patience %d: %s."
                  % (p, ", ".join(vs[r][("a_star_minus_b", "03", m)]["verdict"] for m in ("auroc", "acer_dev"))))
    l3.append("Verdicts that differ between the two runs: %d of 84: %s." % (len(diff), "; ".join(
        "%s stratum %s %s (%s -> %s)" % (COMP_L[cn], st, m, vs[0][(cn, st, m)]["verdict"], vs[1][(cn, st, m)]["verdict"]) for cn, st, m in diff)))
    return l1, l2, l3


class AblBase(Base):
    """Run di ablazione delle sessioni, run con pazienza 12 e loro confronto: utilità comuni."""

    def fault(self, edit, needle):
        fx = Fixture()
        edit(fx)
        self.build_fails(fx, needle)

    def name(self, i):
        a, k, s = self.fx.abl_models[i]
        return "%s_fold%s_seed%s" % (a, k, s)

    def label(self, i):
        a, k, s = self.fx.abl_models[i]
        return "%s fold %s seed %s" % (ARM_L[a], k, s)

    def unchanged(self):
        return [i for i, m in enumerate(self.fx.abl_models) if not self.fx.changed(m)]

    def verdicts(self, fx, run):
        return {(c, st, m): fx.abl[run]["agg"]["comparisons"][c]["verdicts"][st][m] for c, _, _, _ in COMPS for st in STRATA4 for m in KEYS7}


class TestAblationRows(AblBase):
    """Righe della tabella unica e annessi del run di ablazione e della verifica della pazienza, cella per cella."""

    def test_rows_cell_by_cell(self):
        doc = S.build(self.research())
        for run, p in ((0, 4), (1, 12)):
            rows = [r for r in doc["rows"] if r["block"] == "Session ablation run, patience %d" % p]
            ids = {"arm_b": "B", "arm_a_star": "Astar", "arm_b_prime": "Bprime"}
            self.assertEqual([r["row"] for r in rows], ["SESSABL-P%d-%s-seed%s" % (p, ids[a], s) for a in ARMS for s in SEEDS])
            agg = self.fx.abl[run]["agg"]
            for r, (a, s) in zip(rows, [(a, s) for a in ARMS for s in SEEDS]):
                m, b = agg["pooled_per_arm_seed"][a][s]["all"], agg["bootstrap_per_arm_seed"][a][s]["all"]
                self.assertEqual((r["n_bona_fide"], r["n_attack"], r["rejected_bona_fide"], r["accepted_attacks"], r["rejected_bona_fide_dev"],
                                  r["accepted_attacks_dev"]),
                                 (5105, 7509, m["rejected_bona_fide_05"], m["accepted_attacks_05"], m["rejected_bona_fide_dev"], m["accepted_attacks_dev"]))
                self.assertEqual((r["apcer"], r["bpcer"], r["acer"], r["auroc"], r["apcer_dev"], r["bpcer_dev"], r["acer_dev"]),
                                 tuple(f4(m[x]) for x in ("apcer_05", "bpcer_05", "acer_05", "auroc", "apcer_dev", "bpcer_dev", "acer_dev")))
                self.assertEqual(r["bootstrap"], "ACER 0.5 %s-%s; ACER dev %s-%s; AUROC %s-%s"
                                 % tuple(f4(b[k][e]) for k in ("acer_05", "acer_dev", "auroc") for e in ("low", "high")))
                cards = [agg["training"]["%s_fold%s_seed%s" % (a, k, s)] for k in "1234"]
                def sp(v):
                    return ("%d" % min(v)) if min(v) == max(v) else "%d-%d" % (min(v), max(v))
                self.assertEqual(r["model"], "LivenessNet, %s, the 4 fold models of seed %s pooled over the held-out folds; early-stopping "
                                             "patience %d (epochs run %s, restored epochs %s)"
                                 % (S.ARM_TEXTS[a].split(":")[0], s, p, sp([c["epochs_run"] for c in cards]),
                                    sp([c["restored_epoch"] for c in cards])))
                if run == 1:
                    self.assertIn("(epochs run 12, restored epochs", r["model"])
                self.assertEqual((r["test_set"], r["threshold"], r["threshold_dev"], r["run_id"], r["transfer"], r["extractor"]),
                                 (doc["texts"]["folds_test_all"], "0.5", S.THR_DEV_POOLED, self.fx.abl_ids[run][0], S.TRANSFER_ABL, S.EXTRACTOR_FOLDS))
                self.assertEqual(r["early_stopping"], "common dev of the fold (sessions 01-02 of the subjects of the next held-out fold, no "
                                                      "session 03): 400 frames, 100 per class x session cell (01/bf, 01/att, 02/bf, 02/att); "
                                                      "early stopping with patience %d, at most 12 epochs, minimum dev val_loss restored; "
                                                      "learning-rate schedule; EER threshold" % p)
                pre = ("sessabl", "patience")[run]
                self.assertEqual([x.split(" (")[0] for x in r["source"].split("; ")],
                                 [S.SOURCES[pre + "_aggregate"], S.SOURCES[pre + "_record"], S.SOURCES["folds_schema"]])
        self.assertEqual(doc["texts"]["abl_train_arm_b"], "NUAA, arm B, without session 03: sessions 01-02 of the fitting subjects with frames in "
                                                          "sessions 01-02: 1,400 frames per fold, 350/350/350/350 per class x session cell "
                                                          "(01/bf, 01/att, 02/bf, 02/att; deterministic spread rule)")
        self.assertIn(": 1,400 frames per fold, 234/233/233/234/233/233 per class x session cell (01/bf, 01/att, 02/bf, 02/att, 03/bf, 03/att;",
                      doc["texts"]["abl_train_arm_a_star"])
        self.assertIn(": 934 frames per fold, 234/233/233/234 per class x session cell", doc["texts"]["abl_train_arm_b_prime"])
        self.assertEqual(doc["session_ablation"]["runs"]["new"]["amendment"]["to"], 12)

    def test_compact_rows_and_legend(self):
        text = S.outputs(S.build(self.research()))["summary_table_compact.md"]
        m = self.fx.abl[1]["agg"]["pooled_per_arm_seed"]["arm_a_star"]["43"]["all"]
        b = self.fx.abl[1]["agg"]["bootstrap_per_arm_seed"]["arm_a_star"]["43"]["all"]
        self.assertIn("| SESSABL-P12 | LivenessNet, arm A*, 4 fold models, seed 43 (pooled) | T-ABL-A* | S-FOLD-ALL | E-CACHE | P-REF | 0.5 (fixed) "
                      "| %d / %d | %s | %s | %s | %s | 4 fold values (EER, common dev) | %d / %d | %s | %s | %s | %s-%s; %s-%s; %s-%s | %s | "
                      "subject-held-out (session ablation) |"
                      % (m["rejected_bona_fide_05"], m["accepted_attacks_05"], f4(m["apcer_05"]), f4(m["bpcer_05"]), f4(m["acer_05"]), f4(m["auroc"]),
                         m["rejected_bona_fide_dev"], m["accepted_attacks_dev"], f4(m["apcer_dev"]), f4(m["bpcer_dev"]), f4(m["acer_dev"]),
                         f4(b["acer_05"]["low"]), f4(b["acer_05"]["high"]), f4(b["acer_dev"]["low"]), f4(b["acer_dev"]["high"]),
                         f4(b["auroc"]["low"]), f4(b["auroc"]["high"]), self.fx.abl_ids[1][0]), text)
        self.assertIn("| SESSABL-P4 | LivenessNet, arm B', 4 fold models, seed 44 (pooled) | T-ABL-B' |", text)
        for code in ("T-ABL-B", "T-ABL-A*", "T-ABL-B'", "(EER, common dev)", "subject-held-out (session ablation)"):
            self.assertIn("- `%s` (" % code, text)
        self.assertIn("SESSABL-P4 and SESSABL-P12 (session ablation run", text)
        self.assertIn("patience 4 in the SESSABL-P4 rows and 12 in the SESSABL-P12 rows", text)
        self.assertIn("Session_ablation_run `20261106-SESSABLRUN-d999458`, Patience_12_run `20261109-SESSABLRUN-6a50265`, "
                      "Patience_comparison `20261109-PATVER-4f2b925`.", text)

    def test_ablation_annex_cell_by_cell(self):
        text = S.outputs(S.build(self.research()))["session_ablation_table.md"]
        agg = self.fx.abl[1]["agg"]
        m, b = agg["pooled_per_arm_seed"]["arm_a_star"]["43"]["02"], agg["bootstrap_per_arm_seed"]["arm_a_star"]["43"]["02"]
        self.assertIn("| A* | 43 | 02 | 12 | %d | %d | 9 | %d / %d | %s | %s | %s [%s-%s] | %d / %d | %s | %s | %s [%s-%s] | %s [%s-%s] |"
                      % (m["n_bona_fide"], m["n_attack"], m["rejected_bona_fide_05"], m["accepted_attacks_05"], f4(m["bpcer_05"]), f4(m["apcer_05"]),
                         f4(m["acer_05"]), f4(b["acer_05"]["low"]), f4(b["acer_05"]["high"]), m["rejected_bona_fide_dev"], m["accepted_attacks_dev"],
                         f4(m["bpcer_dev"]), f4(m["apcer_dev"]), f4(m["acer_dev"]), f4(b["acer_dev"]["low"]), f4(b["acer_dev"]["high"]),
                         f4(m["auroc"]), f4(b["auroc"]["low"]), f4(b["auroc"]["high"])), text)
        e = self.fx.abl[0]["agg"]["pooled_summary_over_seeds"]["arm_b_prime"]["all"]
        self.assertIn("| B' | all | 4 | %s |" % " | ".join("%s (%s-%s)" % (f4(e[k]["median"]), f4(e[k]["min"]), f4(e[k]["max"]))
                                                         for k in ("acer_05", "acer_dev", "auroc")), text)
        vp, vn = self.verdicts(self.fx, 0), self.verdicts(self.fx, 1)
        changed = [k for k in vp if (vp[k]["verdict"], vp[k]["direction"]) != (vn[k]["verdict"], vn[k]["direction"])]
        self.assertTrue(0 < len(changed) < 84)
        self.assertEqual(len(changed), self.fx.verdicts_changed)
        lines = [l for l in text.splitlines() if l.startswith("| A* - B") or l.startswith("| B' - B")]
        self.assertEqual(len(lines), 84)
        self.assertEqual(sum(l.endswith("| yes |") for l in lines), len(changed))
        for (c, st, m) in changed:
            self.assertTrue(any(l.startswith("| %s | %s | %s |" % (COMP_L[c], st, m)) and l.endswith("| yes |") for l in lines), (c, st, m))
        c, st, m = next(k for k in vn if vn[k]["verdict"] == "consistent")
        pps = self.fx.abl[1]["agg"]["comparisons"][c]["pooled_per_seed"]
        cell = "; ".join("%s [%s, %s]" % (f4(pps[s][st][m]["difference"]), f4(pps[s][st][m]["low"]), f4(pps[s][st][m]["high"])) for s in SEEDS)
        self.assertTrue(any(l.startswith("| %s | %s | %s |" % (COMP_L[c], st, m)) and "| consistent (%s) | %s |" % (vn[(c, st, m)]["direction"], cell) in l
                            for l in lines))
        counts = [sum(v["verdict"] == "consistent" for v in d.values()) for d in (vp, vn)]
        self.assertIn("- Verdicts: 84 per run (3 comparisons x 4 strata x 7 metrics); patience 4: consistent %d, inconclusive %d; patience 12: "
                      "consistent %d, inconclusive %d." % (counts[0], 84 - counts[0], counts[1], 84 - counts[1]), text)
        self.assertIn("- Verdicts that differ between the two runs: %d of 84: " % len(changed), text)
        self.assertEqual(len([l for l in text.splitlines() if l.startswith("| ") and l[2:4] in ("B ", "A*", "B'") and l.count("|") == 17]), 72)

    def test_patience_table_cell_by_cell(self):
        text = S.outputs(S.build(self.research()))["patience_table.md"]
        rows = [l for l in text.splitlines() if l.startswith("| ") and l[2:4] in ("B ", "A*", "B'")]
        self.assertEqual(len(rows), 72)
        t1, t2 = rows[:36], rows[36:]
        for i, (a, k, s) in enumerate(self.fx.abl_models):
            P, N = self.fx.sides[(a, k, s)]
            cells = [ARM_L[a], k, s, "yes" if i in self.fx.ten else "no"]
            for x in (P, N):
                cells += [str(x["epochs_run"]), str(x["restored_epoch"]), "yes" if x["degenerate"] else "no",
                          "—" if x["first_recovery"] is None else str(x["first_recovery"]), "yes" if x["holding_last"] else "no",
                          "%.4f" % x["best_val_loss"], "%.4f" % x["dev_acc_at_checkpoint"], "%.4f" % x["dev_auroc_at_checkpoint"],
                          str(x["epochs_at_half"]), LR_TEXT.get(x["lr_last"], "?")]
            ch = "yes" if self.fx.changed((a, k, s)) else "no"
            cells += [str(self.fx.hist[(a, k, s)]["npp"]), "yes", ch]
            self.assertEqual(t1[i], "| " + " | ".join(cells) + " |")
            test = [ARM_L[a], k, s, "yes" if i in self.fx.ten else "no", ch]
            for run in (0, 1):
                for st in ("03", "all"):
                    m = self.fx.abl[run]["agg"]["per_arm_fold_seed"][a][k][s][st]
                    test += [fna(m["acer_05"]), fna(m["acer_dev"]), fna(m["auroc"])]
            self.assertEqual(t2[i], "| " + " | ".join(test) + " |")
        self.assertIn("| 0.0000625 |", text)  # lr 6,25e-05 in decimali fissi
        self.assertIn("| B | 4 | 42 | no | yes | NA | NA | NA |", text)  # fold 4 senza bona fide nella sessione 03

    def test_patience_compact_table(self):
        files = S.outputs(S.build(self.research()))
        text, full = files["patience_table_compact.md"], files["patience_table.md"]
        self.assertTrue(text.startswith("# Early-stopping patience 4 and 12, the 36 models in compact form (all columns, sources and "
                                        "checks in `patience_table.md`)\n\n| model | in the ten | restored epoch p4 → p12 | first recovery p4 → p12 | "
                                        "recovery holds at last epoch p4 → p12 | dev acc at ckpt p4 → p12 | restored epoch changed |\n|---|---|---|---|---|---|---|\n"))
        rows = [l for l in text.splitlines() if l.startswith("| ") and not l.startswith("| model")]
        self.assertEqual(len(rows), 36)
        for i, (a, k, s) in enumerate(self.fx.abl_models):
            P, N = self.fx.sides[(a, k, s)]
            fr = lambda x: "—" if x["first_recovery"] is None else str(x["first_recovery"])
            yn = lambda b: "yes" if b else "no"
            self.assertEqual(rows[i], "| %s | %s | %d → %d | %s → %s | %s → %s | %.4f → %.4f | %s |"
                             % (self.label(i), "yes" if i in self.fx.ten else "—", P["restored_epoch"], N["restored_epoch"], fr(P), fr(N),
                                yn(P["holding_last"]), yn(N["holding_last"]), P["dev_acc_at_checkpoint"], N["dev_acc_at_checkpoint"],
                                "yes" if self.fx.changed((a, k, s)) else "—"))
        summ = [l for l in text.splitlines() if l.startswith("- ")]
        self.assertEqual(summ, ["- " + l for l in expected_lines(self.fx)[0]])
        self.assertEqual(summ, [l for l in full.split("## Test")[0].splitlines() if l.startswith("- ")])
        self.assertIn("\nLegend. p4 = run `20261106-SESSABLRUN-d999458` (early-stopping patience 4), p12 = run `20261109-SESSABLRUN-6a50265` "
                      "(patience 12); in the ten = one of the ten degenerate trainings of the p4 run (— = not among the ten);", text)
        self.assertEqual(S.synthesis_lines(S.build(self.research()))["patience compact table"],
                         [l for l in text.splitlines() if l.startswith("|") or l.startswith("- ")])

    def test_patience_summary_lines(self):
        """Ogni riga di riepilogo, per intero, contro i valori della fixture (casi limite compresi)."""
        files = S.outputs(S.build(self.research()))
        l1, l2, l3 = expected_lines(self.fx)
        self.assertEqual([l for l in files["patience_table.md"].split("## Test")[0].splitlines() if l.startswith("- ")], ["- " + l for l in l1])
        self.assertEqual([l for l in files["patience_table.md"].split("## Test")[1].splitlines() if l.startswith("- ")], ["- " + l for l in l2])
        self.assertEqual([l for l in files["session_ablation_table.md"].splitlines() if l.startswith("- ")], ["- " + l for l in l3])
        # la fixture copre i casi limite delle righe: una ripresa all'epoca 10 esatta, un'accuratezza 0,9 esatta, celle NA,
        # conteggi dei verdetti diversi fra le due run
        self.assertIn("at epoch 10 or later with patience 12: 1 of the ten (cut-off of CONCLUSION.md)", l1[12])
        self.assertEqual(l1[-2], "Dev accuracy at the checkpoint below 0.9 with patience 12, the ten: 1 (cut-off of CONCLUSION.md).")
        self.assertEqual(l2[5], "Stratum 03, ACER at 0.5, patience 12 vs 4: NA in 9 of 36 (a class absent).")
        self.assertTrue(all(len("- " + l) <= 120 for l in l1 + l2))
        self.assertNotEqual(l3[0].split("patience 4: ")[1].split(";")[0], l3[0].split("patience 12: ")[1].rstrip("."))


class TestAblationChecks(AblBase):
    """Ogni controllo sulle due run di ablazione con una sorgente guasta."""

    def test_clean_tree_status_and_run_ids(self):
        for run in (0, 1):
            for part in ("agg", "rec", "status"):
                for key, value, needle in (("dirty", True, "dirty is not false"), ("status", "FAIL", "is not PASS"),
                                           ("delivered_configuration", False, "not a run of the delivered configuration"),
                                           ("command", "check-inputs", "not a run of the delivered configuration"),
                                           ("commit", "0000000", "does not end with the recorded commit")):
                    self.fault(lambda fx: fx.abl[run][part].__setitem__(key, value), needle)
            self.fault(lambda fx: fx.abl[run]["status"].__setitem__("passed", False), "status: passed is not true")
            self.fault(lambda fx: [fx.abl[run][p].update(run_id=fx.abl_ids[run][0] + "-dev", commit="dev") for p in ("agg", "rec", "status")],
                       "ends with -dev (trial run)")

    def test_identity_and_anchors(self):
        self.fault(lambda fx: fx.abl[0]["rec"].__setitem__("config_sha256", "9" * 64), "session ablation run run record: config_sha256")
        self.fault(lambda fx: fx.abl[1]["status"].__setitem__("finished", "x"), "patience verification run status: finished")
        self.fault(lambda fx: fx.abl[1]["status"].__setitem__("wall_seconds", 1.0), "status: wall_seconds differs from the run record")
        self.fault(lambda fx: fx.abl[0]["status"].__setitem__("public_sums_sha256", "1" * 64), "public_sums_sha256 is not the SHA-256 of nuaa-session-ablation-run")
        for key, needle in (("sessabl_aggregate", "aggregate.json in SHA256SUMS is not the SHA-256 of nuaa-session-ablation-run/tables/aggregate.json"),
                            ("patience_record", "run_record.json in SHA256SUMS is not the SHA-256 of nuaa-patience-verification/tables/run_record.json")):
            research = self.research()
            p = research / S.SOURCES[key]
            p.write_text(p.read_text() + " ", encoding="utf-8")
            self.build_fails(self.fx, needle, research)
        research = self.research()
        p = research / S.SOURCES["sessabl_sums"]
        p.write_text(p.read_text().replace("  aggregate.md", " aggregate.md"), encoding="utf-8")
        self.build_fails(self.fx, "nuaa-session-ablation-run/tables/SHA256SUMS line 2: not '<sha256>  <file>'", research)

    def test_amendment_of_the_patience_run(self):
        for key, value in (("parent_config_sha256", "5" * 64), ("changed_field", "training.lr"), ("from", 5), ("to", 11)):
            def edit(fx, key=key, value=value):
                for p in ("agg", "rec", "status"):
                    fx.abl[1][p]["amendment"][key] = value
            self.fault(edit, "amendment of the patience-12 run")
        self.fault(lambda fx: [fx.abl[0][p].__setitem__("amendment", {"from": 4}) for p in ("agg", "rec", "status")], "the parent run carries an amendment")
        self.fault(lambda fx: fx.abl[1]["agg"].pop("amendment"), "amendment of the patience-12 run")
        self.fault(lambda fx: fx.abl[1]["rec"]["amendment"].__setitem__("to", 13), "run record: amendment")

    def test_models_complete_and_cap(self):
        self.fault(lambda fx: fx.abl[0]["agg"]["training"].pop("arm_b_fold1_seed42"), "training of arm_b_fold1_seed42 missing")
        self.fault(lambda fx: fx.abl[1]["agg"].__setitem__("completed", 35), "35 of 36 trainings complete")
        self.fault(lambda fx: fx.abl[1]["agg"].__setitem__("missing", ["arm_b_fold1_seed42"]), "trainings complete, missing ['arm_b_fold1_seed42']")
        self.fault(lambda fx: fx.abl[0]["rec"]["completed"].pop(), "the models of the aggregate, completed and training_order are not the 36")
        self.fault(lambda fx: fx.abl[1]["rec"].__setitem__("stopped", {"cause": "cap"}), "stopped {'cause': 'cap'}")
        self.fault(lambda fx: fx.abl[1]["rec"].__setitem__("final_writes_after_cap", True), "final writes after the cap True")
        self.fault(lambda fx: fx.abl[1]["rec"].__setitem__("wall_seconds", 21600.5), "wall time 21600.5 s above the cap 21600 s")
        self.fault(lambda fx: fx.abl[0]["rec"]["timings"].pop("train arm_a_star_fold3_seed44"), "no timing for train arm_a_star_fold3_seed44")
        self.fault(lambda fx: fx.abl[0]["rec"].__setitem__("train_seeds", [42, 43]), "seeds [42, 43] differ from the folds run")

    def test_folds_anchors_schema_and_bootstrap(self):
        self.fault(lambda fx: fx.abl[0]["agg"]["reference_a"].__setitem__("aggregate_sha256", "2" * 64), "reference A (run id, aggregate SHA-256) differs")
        self.fault(lambda fx: fx.abl[1]["rec"]["checks"]["folds_run"].__setitem__("run_record_json_sha256", "2" * 64),
                   "folds run (run id, aggregate and record SHA-256) differs")
        self.fault(lambda fx: fx.abl[1]["rec"]["checks"]["schema"].__setitem__("run_id", "20261103-FOLDS-0000000"), "fold schema (run id")
        self.fault(lambda fx: fx.abl[0]["rec"]["checks"]["schema"]["test_per_fold"].reverse(), "fold schema (run id, test rows per fold")
        self.fault(lambda fx: [fx.abl[0][p]["bootstrap"].__setitem__("resamples", 999) for p in ("agg", "rec")],
                   "bootstrap parameters differ from the folds run")
        self.fault(lambda fx: fx.abl[1]["rec"]["bootstrap"].__setitem__("recipe", "other"), "run record: bootstrap differs from the aggregate")
        self.fault(lambda fx: fx.abl[0]["agg"]["reference_a"].__setitem__("run_id", "20261103-FOLDSRUN-0000000"), "reference A")

    def test_cells_and_cards(self):
        self.fault(lambda fx: fx.abl[0]["agg"]["training"]["arm_b_prime_fold2_seed43"].__setitem__("n_fitting", 935),
                   "arm_b_prime_fold2_seed43: 935 fitting / 400 dev frames, the cells of the record give 934 / 400")
        self.fault(lambda fx: fx.abl[1]["rec"]["checks"]["cells"]["counts"]["arm_a_star"].__setitem__("03/attack", 232),
                   "cells differ from the session ablation run")
        self.fault(lambda fx: fx.abl[1]["rec"]["checks"]["cells"]["dev"].__setitem__("02/attack", 0), "a cell count is not a positive integer")
        self.fault(lambda fx: fx.abl[0]["agg"]["training"]["arm_b_fold4_seed44"].__setitem__("restored_epoch", 9), "restored epoch 9 outside 1..6")
        self.fault(lambda fx: fx.abl[0]["agg"]["training"]["arm_b_fold4_seed44"].__setitem__("fold", 3), "arm, fold or seed of the card differ")
        self.fault(lambda fx: fx.abl[1]["agg"]["training"]["arm_b_fold4_seed44"].pop("dev_threshold"), "training field dev_threshold of arm_b_fold4_seed44 missing")

    def test_metrics_rates_denominators_and_sums(self):
        def rate(fx):
            fx.abl[1]["agg"]["per_arm_fold_seed"]["arm_a_star"]["2"]["43"]["03"]["accepted_attacks_dev"] += 1  # tassi invariati
        self.fault(rate, "A* fold 2 seed 43 stratum 03 at dev: apcer")

        def denom(fx):
            for run in (0, 1):
                m = fx.abl[run]["agg"]["per_arm_fold_seed"]["arm_b"]["3"]["42"]["01"]
                m["n_attack"] += 1
                refresh(m)
        self.fault(denom, "B fold 3 seed 42 stratum 01: denominators 280/181 differ from the fold schema 280/180")

        def notsum(fx):
            m = fx.abl[0]["agg"]["per_arm_fold_seed"]["arm_b_prime"]["4"]["44"]["all"]
            m["rejected_bona_fide_dev"] += 1
            refresh(m)
        self.fault(notsum, "B' fold 4 seed 44: stratum all at dev is not the sum of the session strata")

        def pooled(fx):
            m = fx.abl[1]["agg"]["pooled_per_arm_seed"]["arm_b"]["44"]["02"]
            m["accepted_attacks_05"] += 1
            refresh(m)
        self.fault(pooled, "B seed 44 stratum 02 at 05: pooled counts are not the sum of the 4 folds")
        self.fault(lambda fx: fx.abl[0]["agg"]["pooled_per_arm_seed"]["arm_a_star"]["42"]["01"].__setitem__("n_subjects", 8), "8 pooled subjects, the folds run has 9")
        self.fault(lambda fx: fx.abl[0]["agg"]["per_arm_fold_seed"]["arm_a_star"]["1"]["42"]["all"].__setitem__("n_subjects", 5), "5 subjects, the fold schema has 4")
        self.fault(lambda fx: fx.abl[1]["agg"]["pooled_summary_over_seeds"]["arm_b"]["all"]["auroc"].__setitem__("median", 0.1),
                   "summary B stratum all auroc: median or range")
        self.fault(lambda fx: fx.abl[1]["agg"]["pooled_summary_over_seeds"]["arm_b"]["03"]["acer_dev"]["values"].reverse(),
                   "summary B stratum 03 acer_dev: missing or values differ")

        def rev(fx):
            b = fx.abl[0]["agg"]["bootstrap_per_arm_seed"]["arm_b_prime"]["43"]["03"]["acer_05"]
            b["low"], b["high"] = b["high"] + 0.1, b["low"]
        self.fault(rev, "bootstrap B' seed 43 stratum 03: acer_05 interval missing or reversed")
        self.fault(lambda fx: fx.abl[1]["agg"]["bootstrap_per_arm_seed"]["arm_b"]["42"]["all"].__setitem__("seed", 1),
                   "bootstrap B seed 42 stratum all: subjects or parameters")
        self.fault(lambda fx: fx.abl[1]["agg"]["bootstrap_per_arm_seed"]["arm_b"]["42"].pop("01"), "bootstrap B seed 42 stratum 01 missing")
        self.fault(lambda fx: fx.abl[0]["agg"].__setitem__("strata", ["01", "02", "03"]), "strata")

    def test_comparisons_and_verdicts(self):
        self.fault(lambda fx: fx.abl[0]["agg"]["comparisons"]["b_prime_minus_b"]["per_fold_seed"]["2"]["44"]["03"].__setitem__("auroc", 0.5),
                   "comparison B' - B: 1 per-model differences are not minuend minus subtrahend (first: B' fold 2 seed 44 03 auroc)")
        self.fault(lambda fx: fx.abl[1]["agg"]["comparisons"]["a_star_minus_b"]["pooled_per_seed"]["43"]["all"]["acer_05"].__setitem__("difference", 0.5),
                   "comparison A* - B seed 43 stratum all acer_05: the difference is not that of the pooled rows")
        self.fault(lambda fx: fx.abl[1]["agg"]["comparisons"]["a_star_minus_b"].__setitem__("role", "secondary"), "minuend, subtrahend or role differ")
        self.fault(lambda fx: fx.abl[1]["agg"]["comparisons"].pop("a_star_minus_b_prime"), "comparisons")
        vn = self.verdicts(self.fx, 1)
        c, st, m = next(k for k in vn if vn[k]["verdict"] == "consistent")
        ci = next(i for i, x in enumerate(COMPS) if x[0] == c)
        for key, value in (("verdict", "inconclusive"), ("direction", "%s higher X than %s" % (ARM_L[COMPS[ci][2]], ARM_L[COMPS[ci][1]])),
                           ("reason", REASONS["sign"]), ("label", "primary")):
            self.fault(lambda fx: fx.abl[1]["agg"]["comparisons"][c]["verdicts"][st][m].__setitem__(key, value),
                       "comparison %s stratum %s %s: verdict" % (COMP_L[c], st, m))
        vi = next(k for k in vn if vn[k]["verdict"] == "inconclusive")
        self.fault(lambda fx: fx.abl[1]["agg"]["comparisons"][vi[0]]["verdicts"][vi[1]][vi[2]].__setitem__("verdict", "consistent"),
                   "comparison %s stratum %s %s: verdict 'consistent'" % (COMP_L[vi[0]], vi[1], vi[2]))
        self.fault(lambda fx: fx.abl[0]["agg"]["primary_verdicts"]["auroc"].__setitem__("verdict", "other"), "primary verdicts differ")

    def test_reading_rule_boundaries(self):
        """Un intervallo che tocca 0 non lo esclude; una differenza nulla o un seed senza intervallo non danno «consistent»."""
        vn = self.verdicts(self.fx, 1)
        c, st, m = next(k for k in vn if vn[k]["verdict"] == "consistent")
        pos = self.fx.abl[1]["agg"]["comparisons"][c]["pooled_per_seed"]["42"][st][m]["difference"] > 0

        def touch(fx):
            fx.abl[1]["agg"]["comparisons"][c]["pooled_per_seed"]["43"][st][m]["low" if pos else "high"] = 0.0
        self.fault(touch, "comparison %s stratum %s %s: verdict 'consistent'" % (COMP_L[c], st, m))
        self.fault(lambda fx: fx.abl[1]["agg"]["comparisons"][c]["pooled_per_seed"]["44"][st][m].pop("high"),
                   "comparison %s stratum %s %s: verdict 'consistent'" % (COMP_L[c], st, m))
        self.assertEqual(S.rule_reason([{"difference": 0.1, "low": 0.0, "high": 0.2}] * 3), "interval")
        self.assertEqual(S.rule_reason([{"difference": -0.1, "low": -0.2, "high": 0.0}] * 3), "interval")
        self.assertEqual(S.rule_reason([{"difference": 0.1, "low": 0.01, "high": 0.2}] * 3), "consistent")
        self.assertEqual(S.rule_reason([{"difference": 0.1, "low": 0.01, "high": 0.2}, {"difference": 0, "low": -1, "high": 1},
                                        {"difference": -0.1, "low": -0.2, "high": -0.01}]), "zero")
        self.assertEqual(S.rule_reason([{"difference": 0.1, "low": 0.01, "high": 0.2}, {"difference": -0.1, "low": -0.2, "high": -0.01},
                                        {"difference": 0.1, "low": 0.01, "high": 0.2}]), "sign")
        self.assertEqual(S.rule_reason([{"difference": 0.1, "low": 0.01, "high": 0.2}, None, {"difference": 0.1, "low": 0.01, "high": None}]),
                         "missing")
        self.fault(lambda fx: fx.abl[0]["agg"].__setitem__("reading_rule", ""), "reading rule missing")
        self.fault(lambda fx: fx.abl[1]["agg"].__setitem__("reading_rule", "another rule"), "reading rule differs from the session ablation run")


class TestPatienceChecks(AblBase):
    """Il confronto delle due run: ogni controllo con una sorgente guasta."""

    def test_comparison_run_identity_and_anchors(self):
        self.fault(lambda fx: fx.pv.__setitem__("dirty", True), "patience comparison aggregate: dirty is not false")
        self.fault(lambda fx: fx.pv_status.__setitem__("passed", False), "patience comparison status: passed False")
        self.fault(lambda fx: fx.pv_status.__setitem__("verdict", "negative"), "status: verdict 'negative' differs from the aggregate")
        self.fault(lambda fx: fx.pv.__setitem__("delivered_configuration", False), "not a run of the delivered configuration")
        self.fault(lambda fx: [d.update(run_id="20261109-PATVER-4f2b925-dev", commit="dev") for d in (fx.pv, fx.pv_status)], "ends with -dev (trial run)")
        self.fault(lambda fx: fx.pv["declarations"].__setitem__("no_test_scores_read", False), "declaration no_test_scores_read is not true")
        self.fault(lambda fx: fx.pv_status.__setitem__("public_sums_sha256", "3" * 64), "public_sums_sha256 is not the SHA-256 of nuaa-patience-verification/comparison/SHA256SUMS")
        for key in ("patver_aggregate", "patver_comparison", "patver_curves"):
            research = self.research()
            p = research / S.SOURCES[key]
            p.write_text(p.read_text() + "\n", encoding="utf-8")
            self.build_fails(self.fx, "%s in SHA256SUMS is not the SHA-256" % S.SOURCES[key].rsplit("/", 1)[1], research)

    def test_inputs_anchored_to_both_runs(self):
        for key, field, value in (("parent_run", "status_sha256", "4" * 64), ("verification_run", "public_sums_sha256", "4" * 64),
                                  ("parent_run", "run_id", "20261106-SESSABLRUN-0000000"), ("verification_run", "config_sha256", "4" * 64),
                                  ("parent_run", "complete_models", 35), ("verification_run", "status", "INCOMPLETE"),
                                  ("verification_run", "restricted_sums_sha256", "4" * 64)):
            self.fault(lambda fx: fx.pv["inputs"][key].__setitem__(field, value), "inputs.%s" % key)
        self.fault(lambda fx: fx.pv["inputs"]["verification_run"].__setitem__("early_stopping_patience", 12.0), "is not a positive integer")
        self.fault(lambda fx: fx.pv["inputs"]["verification_run"].__setitem__("early_stopping_patience", 4), "the two runs have the same patience 4")
        self.fault(lambda fx: fx.pv["recipe"].__setitem__("max_epochs", "12"), "recipe.max_epochs '12' is not a positive integer")
        self.fault(lambda fx: fx.pv["criteria"].pop("recovery"), "criteria incomplete")

    def test_rows_rebuilt_from_the_curves(self):
        for field, value in (("parent_restored_epoch", 3), ("new_first_recovery", 2), ("parent_degenerate", False), ("new_epochs_at_half", 9),
                             ("parent_lr_last", 0.1), ("new_best_val_loss", 0.123)):
            self.fault(lambda fx: fx.pv_rows[11].__setitem__(field, value), "comparison.csv B fold 4 seed 44 patience")
        self.fault(lambda fx: fx.pv_rows[5].__setitem__("new_dev_acc_at_checkpoint", 0.5), "dev accuracy at the checkpoint differs")

        def replay(fx):  # epoca minima spostata: l'arresto con pazienza 4 non torna più con le epoche registrate
            for c in fx.pv_curves:
                if (c["arm"], c["fold"], c["seed"], c["epoch"]) == ("arm_b", "2", "43", 2):
                    c["val_loss"] = 0.0001
        self.fault(replay, "early stopping with patience 4 and best epoch 2 gives 6")
        self.fault(lambda fx: fx.pv_curves[30].__setitem__("saturated", not fx.pv_curves[30]["saturated"]), "flags of curves.csv differ")
        self.fault(lambda fx: fx.pv_rows[7].__setitem__("shared_identical", False), "shared history not identical")
        self.fault(lambda fx: fx.pv_rows[7].__setitem__("max_abs_diff_shared_lr", 1e-9), "shared history not identical")
        self.fault(lambda fx: fx.abl[1]["agg"]["training"]["arm_a_star_fold2_seed42"].__setitem__("best_val_loss", 0.5),
                   "differ from the training card of the run")

    def test_csv_structure(self):
        self.fault(lambda fx: fx.pv_rows.reverse(), "expected the models in arm, fold, seed order")
        research = self.research()
        p = research / S.SOURCES["patver_comparison"]
        p.write_text(p.read_text().replace("arm,fold,seed,in_ten", "arm,fold,seed,ten", 1), encoding="utf-8")
        self.build_fails(self.fx, "nuaa-patience-verification/comparison/comparison.csv: columns", research)
        self.fault(lambda fx: fx.pv_rows[3].__setitem__("parent_epochs_run", "12.0"), "'12.0' is not an integer")
        self.fault(lambda fx: fx.pv_rows[3].__setitem__("in_ten", "yes"), "'yes' is not 0 or 1")
        self.fault(lambda fx: fx.pv_curves.__setitem__(slice(0, 12), []), "the models of curves.csv are not the 36 models")
        self.fault(lambda fx: fx.pv_curves.pop(40), "curves.csv epochs are not 1..12")
        self.fault(lambda fx: fx.pv_rows.pop(), "comparison.csv has 35 rows, expected 36")
        self.fault(lambda fx: fx.pv_curves[2].__setitem__("val_loss", "nan"), "'nan' is not a finite number")
        self.fault(lambda fx: fx.pv_rows[0].__setitem__("new_status", "missing"), "status 'missing' in the patience-12 run")

    def test_the_ten_verdict_and_sections(self):
        def flip(fx):
            fx.pv_rows[1]["in_ten"] = True
            for c in fx.pv_curves:
                if (c["arm"], c["fold"], c["seed"]) == ("arm_b", "1", "43"):
                    c["in_ten"] = True
        self.fault(flip, "the models marked in_ten are not the degenerate models of the patience-4 run")
        self.fault(lambda fx: fx.pv["original_degenerate"].pop(), "original_degenerate differs")
        self.fault(lambda fx: fx.pv.__setitem__("models_without_recovery_parent", []), "models_without_recovery_parent differs")
        self.fault(lambda fx: fx.pv["rows"][4]["new"].__setitem__("dev_auroc", 0.1), "aggregate row %s differs from comparison.csv" % self.label(4))
        self.fault(lambda fx: fx.pv["rows"][4].__setitem__("in_ten", 1), "aggregate row %s differs" % self.label(4))
        self.fault(lambda fx: fx.pv["sections"]["a"].__setitem__("identical", 35), "section a (shared histories) differs")
        self.fault(lambda fx: [d.__setitem__("verdict", "negative") for d in (fx.pv, fx.pv["sections"]["b"], fx.pv_status)], "verdict 'negative'")
        self.fault(lambda fx: fx.pv["sections"]["b"].__setitem__("ten_recovered", 9), "section b differs from the rows in ['ten_recovered']")
        self.fault(lambda fx: fx.pv["sections"]["b"]["ten_first_recovery"].__setitem__(self.label(0), 1), "['ten_first_recovery']")
        self.fault(lambda fx: fx.pv["sections"]["c"]["recovered"].__setitem__("parent", 36), "section c recovered counts differ")
        self.fault(lambda fx: fx.pv["sections"]["c"]["summary"][1]["restored_epoch"].__setitem__("median", 1), "section c medians per group differ")
        self.fault(lambda fx: fx.pv["sections"]["d"]["rows"][20]["new"].__setitem__("median", 0.0), "section d rows differ")
        self.fault(lambda fx: fx.pv["sections"]["d"]["primary"][0]["new"].__setitem__("verdict", "other"), "section d primary verdicts differ")
        self.fault(lambda fx: fx.pv["sections"]["d"].__setitem__("verdicts_changed", fx.verdicts_changed + 1),
                   "section d declares %d verdicts changed, the two aggregates give %d" % (self.fx.verdicts_changed + 1, self.fx.verdicts_changed))
        self.fault(lambda fx: fx.pv["sections"]["d"].__setitem__("reported", False), "section d label missing or not reported")
        self.fault(lambda fx: fx.pv["sections"]["e"].__setitem__("new_wall_seconds", 1.0), "section e (wall times, cap, within the cap, stop) differs")
        self.fault(lambda fx: fx.pv["sections"]["e"].__setitem__("within_cap", False), "section e (wall times, cap, within the cap, stop) differs")
        self.fault(lambda fx: fx.pv["sections"]["e"]["per_arm"][4].__setitem__("epochs", 143), "section e per arm")
        self.fault(lambda fx: fx.pv["sections"].pop("e"), "sections a-e missing")

    def test_same_checkpoint_gives_the_same_test_values(self):
        i = self.unchanged()[0]
        self.fault(lambda fx: fx.abl[1]["agg"]["training"][self.name(i)].__setitem__("dev_threshold", 0.77),
                   "%s: same checkpoint in both runs, but test values or dev threshold differ" % self.label(i))
        a, k, s = self.fx.abl_models[i]

        def test_values(fx):
            fx.abl[1]["agg"]["per_arm_fold_seed"][a][k][s]["03"]["auroc"] += 0.001
            fx.abl[1]["agg"]["per_arm_fold_seed"][a][k][s]["all"]["auroc"] += 0.001
        self.fault(test_values, "%s: same checkpoint in both runs" % self.label(i))


class TestReviewFaults(AblBase):
    """Controlli che la revisione avversaria ha tolto senza che un test se ne accorgesse: una sorgente guasta per ciascuno."""

    def consistent(self, run=1):
        v = self.verdicts(self.fx, run)
        return next(k for k in v if v[k]["verdict"] == "consistent")

    def test_dev_run_id_refused_before_the_commit_check(self):
        # il commit resta quello vero: la regola «-dev» deve scattare prima del controllo del suffisso del run id
        self.fault(lambda fx: fx.abl[0]["agg"].__setitem__("run_id", fx.abl_ids[0][0] + "-dev"),
                   "session ablation run aggregate: run id '20261106-SESSABLRUN-d999458-dev' ends with -dev (trial run)")
        self.fault(lambda fx: fx.abl[1]["status"].__setitem__("run_id", fx.abl_ids[1][0] + "-dev"), "status: run id '20261109-SESSABLRUN-6a50265-dev' ends with -dev")
        self.fault(lambda fx: fx.pv_status.__setitem__("run_id", "20261109-PATVER-4f2b925-dev"),
                   "patience comparison status: run id '20261109-PATVER-4f2b925-dev' ends with -dev (trial run)")

    def test_verification_status_of_the_comparison(self):
        self.fault(lambda fx: fx.pv_status.__setitem__("verification_status", "FAIL"),
                   "status: verification_status 'FAIL' is not the status of the patience-verification run 'PASS', PASS")
        self.fault(lambda fx: fx.pv_status.pop("verification_status"), "field 'verification_status' missing")

    def test_direction_metric_word_and_sign(self):
        c, st, m = self.consistent()
        d = self.fx.abl[1]["agg"]["comparisons"][c]["verdicts"][st][m]["direction"]
        other = "AUROC" if m != "auroc" else "ACER (0.5)"
        for new in (d.replace(METRIC_NAME[m], other), d.replace(" higher ", " lower ") if " higher " in d else d.replace(" lower ", " higher "),
                    d.replace(" than ", " than  ")):
            self.fault(lambda fx, new=new: fx.abl[1]["agg"]["comparisons"][c]["verdicts"][st][m].__setitem__("direction", new),
                       "comparison %s stratum %s %s: verdict 'consistent'" % (COMP_L[c], st, m))

    def test_primary_verdicts_direction_reason_and_label(self):
        for key, value in (("direction", "A* higher AUROC than B"), ("reason", REASONS["zero"]), ("label", "secondary")):
            self.fault(lambda fx, key=key, value=value: fx.abl[0]["agg"]["primary_verdicts"]["auroc"].__setitem__(key, value),
                       "primary verdicts differ from the verdicts of A* - B in stratum 03")

    def test_comparison_summary_and_subjects(self):
        self.fault(lambda fx: fx.abl[0]["agg"]["comparisons"]["b_prime_minus_b"]["pooled_summary_over_seeds"]["02"]["acer_dev"].__setitem__("max", 9.0),
                   "comparison B' - B summary stratum 02 acer_dev: median or range is not that of the values")
        self.fault(lambda fx: fx.abl[1]["agg"]["comparisons"]["a_star_minus_b"]["pooled_per_seed"]["44"]["03"].__setitem__("n_subjects", 15),
                   "comparison A* - B seed 44 stratum 03: subjects differ from the pooled rows")

    def test_section_e_train_seconds_tolerance(self):
        self.fault(lambda fx: fx.pv["sections"]["e"]["per_arm"][2].__setitem__("train_seconds", fx.pv["sections"]["e"]["per_arm"][2]["train_seconds"] + 1e-5),
                   "section e per arm (models, epochs, train seconds) differs from the training cards")

    def test_in_ten_of_the_curves(self):
        def flip(fx):
            for c in fx.pv_curves:
                if (c["arm"], c["fold"], c["seed"]) == ("arm_a_star", "1", "42"):
                    c["in_ten"] = not c["in_ten"]
        self.fault(flip, "comparison.csv A* fold 1 seed 42 epoch 1: flags of curves.csv differ from the values")

    def test_negative_verdict_consistent_everywhere_is_refused(self):
        def no_recovery(h):
            h["va"][5:] = [0.6] * 7
        fx = Fixture(overrides={0: no_recovery})
        self.assertEqual((fx.pv["verdict"], fx.pv_status["verdict"], fx.pv["sections"]["b"]["ten_recovered"]), ("negative", "negative", 9))
        self.build_fails(fx, "verdict 'negative' (section b 'negative', status 'negative'); the rows give 'negative'")

    def test_group_sizes_ten_and_36(self):
        def not_degenerate(h):
            h["va"][0] = 0.6
        fx = Fixture(overrides={0: not_degenerate})
        self.assertEqual(len(fx.ten), 9)
        self.build_fails(fx, "9 models in the ten and 36 in all; the groups of section c are the ten, the other 26 and all 36")

    def test_duplicated_sums_line_and_n_dev(self):
        research = self.research()
        p = research / S.SOURCES["patience_sums"]
        p.write_text(p.read_text() + p.read_text().splitlines()[0] + "\n", encoding="utf-8")
        self.build_fails(self.fx, "nuaa-patience-verification/tables/SHA256SUMS: aggregate.json listed twice", research)
        self.fault(lambda fx: fx.abl[1]["agg"]["training"]["arm_b_fold1_seed43"].__setitem__("n_dev", 399),
                   "arm_b_fold1_seed43: 1400 fitting / 399 dev frames, the cells of the record give 1400 / 400")

    def test_same_configuration_in_both_runs(self):
        def same(fx):
            for part in ("agg", "rec", "status"):
                fx.abl[1][part]["config_sha256"] = fx.abl_ids[0][2]
            fx.pv["inputs"]["verification_run"]["config_sha256"] = fx.abl_ids[0][2]
        fx = Fixture()
        same(fx)
        with self.assertRaises(S.SynthesisError) as cm:
            S.build(fx.write(self.root))
        self.assertEqual(str(cm.exception), "consistency problems:\n  - patience comparison: the two runs have the same configuration")


class TestAblationCommands(AblBase):
    def run_main(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = S.main(list(argv))
        return code, out.getvalue()

    def test_duplicated_markdown_rows_are_refused(self):
        self.assertEqual(S.duplicate_rows({"a.md": "| x |\n|---|\n| 1 |\n| 2 |\n\n| y |\n|---|\n| 1 |\n", "b.csv": "1\n1\n"}), [])
        self.assertEqual(S.duplicate_rows({"a.md": "| x |\n|---|\n| 1 |\n| 1 |\n"}), ["a.md: duplicated row | 1 |"])
        doc = S.build(self.research())
        S.checked_outputs(doc)
        doc["session_ablation"]["annex"]["pooled"].append(list(doc["session_ablation"]["annex"]["pooled"][5]))
        with self.assertRaises(S.SynthesisError) as cm:
            S.checked_outputs(doc)
        self.assertIn("duplicated Markdown rows", str(cm.exception))
        self.assertIn("session_ablation_table.md: duplicated row", str(cm.exception))

    def test_check_new_annexes_byte_exact(self):
        research = self.research()
        out = self.root / "tables"
        code, text = self.run_main("build", "--research", str(research), "--out", str(out))
        self.assertEqual(code, 0, text)
        self.assertIn("57 rows", text)
        self.assertIn("36 patience model rows, 72 session-ablation pooled rows, 84 verdict rows", text)
        p = out / "patience_table.md"
        p.write_bytes(p.read_bytes().replace(b"| yes |", b"| no |", 1))
        (out / "session_ablation_table.md").unlink()
        code, text = self.run_main("check", "--research", str(research), "--out", str(out))
        self.assertEqual(code, 1)
        self.assertIn("patience_table.md: differs", text)
        self.assertIn("session_ablation_table.md: missing", text)

    def test_error_paths_without_traceback(self):
        for edit, needle in ((lambda r: (r / S.SOURCES["patver_curves"]).unlink(), "ERROR: source patver_curves missing"),
                             (lambda r: (r / S.SOURCES["patience_status"]).write_text("[1, 2]", encoding="utf-8"), "ERROR: patience_status"),
                             (lambda r: (r / S.SOURCES["sessabl_aggregate"]).write_text("{not json", encoding="utf-8"),
                              "ERROR: a source has an unexpected structure"),
                             (lambda r: (r / S.SOURCES["patver_comparison"]).write_bytes(b"\xff\xfe"), "ERROR: a source has an unexpected structure")):
            research = self.research()
            edit(research)
            code, text = self.run_main("build", "--research", str(research), "--out", str(self.root / "t"))
            self.assertEqual(code, 1, text)
            self.assertIn("RESULT: FAIL", text)
            self.assertNotIn("Traceback", text)
            self.assertFalse((self.root / "t").exists())
            if needle != "ERROR: patience_status":
                self.assertIn(needle, text)
        fx = Fixture()
        fx.abl[0]["agg"]["comparisons"]["a_star_minus_b"] = ["not", "an", "object"]
        code, text = self.run_main("build", "--research", str(fx.write(self.root)), "--out", str(self.root / "t"))
        self.assertEqual(code, 1)
        self.assertIn("ERROR: a source has an unexpected structure", text)
        self.assertFalse((self.root / "t").exists())


class TestCommands(Base):
    def run_main(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = S.main(list(argv))
        return code, out.getvalue()

    def test_build_then_check_pass_and_tamper_fails(self):
        research = self.research()
        out = self.root / "tables"
        code, text = self.run_main("build", "--research", str(research), "--out", str(out))
        self.assertEqual(code, 0)
        self.assertIn("RESULT: PASS", text)
        self.assertEqual(sorted(p.name for p in out.iterdir()), sorted(S.outputs(S.build(research))))
        for p in out.iterdir():
            self.assertNotIn(b"\r\n", p.read_bytes())
        code, text = self.run_main("check", "--research", str(research), "--out", str(out))
        self.assertEqual((code, "0 problems" in text), (0, True))
        p = out / "summary_table.md"
        p.write_bytes(p.read_bytes().replace(b"| 0.", b"| 1.", 1))
        code, text = self.run_main("check", "--research", str(research), "--out", str(out))
        self.assertEqual(code, 1)
        self.assertIn("summary_table.md: differs", text)
        (out / "summary_table.csv").unlink()
        code, text = self.run_main("check", "--research", str(research), "--out", str(out))
        self.assertIn("summary_table.csv: missing", text)

    def test_check_is_byte_exact(self):
        research = self.research()
        out = self.root / "tables"
        self.run_main("build", "--research", str(research), "--out", str(out))
        p = out / "summary_table.csv"
        p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
        code, text = self.run_main("check", "--research", str(research), "--out", str(out))
        self.assertEqual(code, 1)
        self.assertIn("summary_table.csv: differs", text)

    def test_check_synthesis_lines(self):
        research = self.research()
        out = self.root / "tables"
        self.run_main("build", "--research", str(research), "--out", str(out))
        doc = self.root / "SYNTHESIS.md"
        doc.write_text("# Doc\n\n" + (out / "summary_table_compact.md").read_text() + "\n" + (out / "patience_table_compact.md").read_text(),
                       encoding="utf-8")
        code, text = self.run_main("check", "--research", str(research), "--out", str(out), "--synthesis", str(doc))
        self.assertEqual(code, 0, text)
        self.assertIn("+ synthesis", text)
        full = doc.read_text()
        for victim in ("| B fold 2 seed 43 |", "- Restored epoch changed:", "| model | in the ten |", "|---|---|---|---|---|---|---|\n"):
            lines = full.splitlines(keepends=True)
            idx = next(i for i, l in enumerate(lines) if l.startswith(victim.rstrip("\n")) and (victim != "|---|---|---|---|---|---|---|\n" or l == victim))
            for altered in ("".join(lines[:idx] + lines[idx + 1:]), "".join(lines[:idx] + [lines[idx].rstrip("\n") + " x\n"] + lines[idx + 1:])):
                doc.write_text(altered, encoding="utf-8")
                code, text = self.run_main("check", "--research", str(research), "--out", str(out), "--synthesis", str(doc))
                self.assertEqual(code, 1, victim)
                self.assertIn("1 of 59 lines of the patience compact table are not in the document verbatim", text)
        doc.write_text(full, encoding="utf-8")
        lines = doc.read_text().splitlines()
        idx = next(i for i, l in enumerate(lines) if l.startswith("| N1 | LivenessNet, new CPU training, seed 43"))
        lines[idx] = lines[idx].replace("seed 43", "seed 45")
        doc.write_text("\n".join(lines), encoding="utf-8")
        code, text = self.run_main("check", "--research", str(research), "--out", str(out), "--synthesis", str(doc))
        self.assertEqual(code, 1)
        self.assertIn("1 of 59 lines of the compact table", text)

    def test_build_reports_consistency_error_and_writes_nothing(self):
        self.fx.passed["crop"] = False
        research = self.research()
        code, text = self.run_main("build", "--research", str(research), "--out", str(self.root / "t"))
        self.assertEqual(code, 1)
        self.assertIn("ERROR: ", text)
        self.assertIn("RESULT: FAIL", text)
        self.assertFalse((self.root / "t").exists())

    def test_malformed_source_reports_fail_without_traceback(self):
        research = self.research()
        (research / S.SOURCES["crop"]).write_text("{not json", encoding="utf-8")
        code, text = self.run_main("build", "--research", str(research), "--out", str(self.root / "t"))
        self.assertEqual(code, 1)
        self.assertIn("ERROR: a source has an unexpected structure", text)
        self.assertIn("RESULT: FAIL", text)
        self.assertFalse((self.root / "t").exists())
        research = self.edit_json("n1_aggregate", lambda d: d["seeds"]["42"].pop("training"))
        code, text = self.run_main("build", "--research", str(research), "--out", str(self.root / "t"))
        self.assertEqual(code, 1)
        self.assertIn("RESULT: FAIL", text)


class TestMarkdown(Base):
    def test_md_tables_parsing(self):
        text = "intro\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\ntext\n\n| c |\n|:-:|\n| x |\n| short | row |\n"
        tables = S.md_tables(text)
        self.assertEqual(len(tables), 2)
        self.assertEqual(tables[0][1], [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}])
        self.assertEqual(tables[1][1][0], {"c": "x"})
        self.assertIn("_raw", tables[1][1][1])

    def test_keyed_rows_refuses_duplicates(self):
        rows = [{"k": "a", "v": "1"}, {"_raw": ["x"], "k": "x"}, {"k": "a", "v": "2"}]
        with self.assertRaises(S.SynthesisError) as cm:
            S.keyed_rows(rows, "k", "T")
        self.assertIn("T: duplicate row a", str(cm.exception))
        self.assertEqual(S.keyed_rows(rows[:2], "k", "T"), {"a": {"k": "a", "v": "1"}})

    def test_f4_and_fmt(self):
        self.assertEqual(S.f4(None), "NA")
        self.assertEqual(S.f4(0.76666666), "0.7667")
        self.assertEqual(S.f4("1"), "1.0000")
        self.assertEqual(S.fmt(2507), "2,507")
        self.assertEqual(S.as_int("12", "x"), 12)


if __name__ == "__main__":
    unittest.main(verbosity=2 if "-v" in sys.argv else 1, argv=[sys.argv[0]])
