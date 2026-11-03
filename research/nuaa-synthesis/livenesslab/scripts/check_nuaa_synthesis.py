"""
Test di scripts/nuaa_synthesis.py su un albero research/ sintetico scritto in una cartella temporanea: nessun dato
reale, nessuna dipendenza oltre la libreria standard. La fixture costruisce le otto sorgenti con numeri inventati,
distinti per ogni sottoinsieme e coerenti fra loro; ogni test di coerenza guasta una sorgente per volta e verifica che
build si fermi con il problema giusto; i test di costruzione confrontano ogni riga e ogni annesso con la fixture.

Uso (dalla radice del repository del gruppo): python3 -I research/nuaa-synthesis/livenesslab/scripts/check_nuaa_synthesis.py [-v]
"""
import contextlib
import copy
import csv
import importlib.util
import io
import json
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


def f6(v):
    return "%.6f" % v


class Fixture:
    """Le otto sorgenti sintetiche, modificabili prima della scrittura."""

    def __init__(self):
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
        return {
            S.SOURCES["c1_table"]: self.c1_table(),
            S.SOURCES["m1_table"]: self.m1_table(),
            S.SOURCES["m2_manifests"]: json.dumps(self.m2_json(), indent=1),
            S.SOURCES["n1_aggregate"]: json.dumps(self.n1_json(), indent=1),
            S.SOURCES["n1_subjects"]: json.dumps(self.n1_subjects_json(), indent=1),
            S.SOURCES["n2_sessions"]: json.dumps(self.n2s_json(), indent=1),
            S.SOURCES["n2_combined"]: json.dumps(self.n2c_json(), indent=1),
            S.SOURCES["crop"]: json.dumps(self.crop_json(), indent=1),
        }

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
        self.assertEqual(len(doc["rows"]), 21)
        blocks = [r["block"] for r in doc["rows"]]
        self.assertEqual(blocks, ["C1"] * 4 + ["M1"] * 4 + ["N1"] * 3 + ["N2 sessions"] * 6 + ["N2 combined"] * 2 + ["C1 crop check"] * 2)
        self.assertEqual(len(doc["cross_dataset_rows"]), 16)
        self.assertEqual(len(doc["context_rows"]), 10)
        self.assertEqual(len(doc["subject_sensitivity"]), 4)
        self.assertEqual(len(set(r["row"] for r in doc["rows"])), 21)
        for r in doc["rows"]:
            self.assertTrue(all(r[c] != "" for c in S.COLUMNS), r)
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
                                       "N2_sessions": self.fx.n2s_run, "N2_combined": self.fx.n2c_run, "C1_crop_check": self.fx.crop_run})
        for r in doc["rows"]:
            for part in r["source"].split("; "):
                path, prefix = part[:-1].split(" (")
                self.assertEqual(prefix, doc["sources"][next(k for k, v in S.SOURCES.items() if v == path)]["sha256"][:12])

    def test_outputs_deterministic_and_compact_table(self):
        research = self.research()
        a, b = S.outputs(S.build(research)), S.outputs(S.build(research))
        self.assertEqual(a, b)
        self.assertEqual(set(a), {"summary_table.md", "summary_table.csv", "summary_table.json", "summary_table_compact.md",
                                  "cross_dataset_table.md", "context_rows_table.md", "subject_sensitivity_table.md"})
        compact = [l for l in a["summary_table_compact.md"].splitlines() if l.startswith("| ") and not l.startswith("| block")]
        self.assertEqual(len(compact), 21)
        text = a["summary_table_compact.md"]
        for code in ("T-NUAA", "T-COMB", "S-300", "S-3000", "S-491", "S-03/9", "E-C1", "E-CACHE", "E-WHOLE", "E-CROP", "P-REF", "P-HIST", "same", "session", "session/9", "dataset"):
            self.assertIn("- `%s` (" % code, text)
        d = self.fx.c1["attacknet_v1__nuaa"]
        self.assertIn("| M1 | AttackNet V1 (C1 ckpt 12/09) | T-NUAA | S-300 | E-C1 | P-REF | %s (dev EER) | %d / %d | %s | %s | %s | %s | %s | session |"
                      % (f6(d["thr"]), d["rej_d"], d["acc_d"], f4(d["acc_d"] / 150), f4(d["rej_d"] / 150), f4((d["acc_d"] / 150 + d["rej_d"] / 150) / 2), f4(d["auc"]), S.M1_RUN), text)
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
        self.assertEqual(len(rows), 21)
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
        doc.write_text("# Doc\n\n" + (out / "summary_table_compact.md").read_text(), encoding="utf-8")
        code, text = self.run_main("check", "--research", str(research), "--out", str(out), "--synthesis", str(doc))
        self.assertEqual(code, 0, text)
        self.assertIn("+ synthesis", text)
        lines = doc.read_text().splitlines()
        idx = next(i for i, l in enumerate(lines) if l.startswith("| N1 | LivenessNet, new CPU training, seed 43"))
        lines[idx] = lines[idx].replace("seed 43", "seed 45")
        doc.write_text("\n".join(lines), encoding="utf-8")
        code, text = self.run_main("check", "--research", str(research), "--out", str(out), "--synthesis", str(doc))
        self.assertEqual(code, 1)
        self.assertIn("1 of 23 lines", text)

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
