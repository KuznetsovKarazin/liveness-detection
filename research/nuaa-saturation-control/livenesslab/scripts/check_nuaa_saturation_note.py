"""
Test di nuaa_saturation_note.py: parser, formula del residuo, conteggi contro valori calcolati a mano su una fixture
sintetica scritta in una cartella temporanea (nessun dato, nessun file letto dal disco tranne, se presenti, le sorgenti
pubbliche reali per la prova da capo a fondo), percorsi d'errore, controllo della nota, comandi build e check.

Uso (dalla radice del repository): python3 -I -B research/nuaa-saturation-control/livenesslab/scripts/check_nuaa_saturation_note.py -v
"""
import contextlib
import csv
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nuaa_saturation_note as N  # noqa: E402

REPO = HERE.parents[3]

ARCH_TEXT = '''"""Fixture architectures."""
from tensorflow.keras import layers, Model


class LivenessNet(BaseArchitecture):
    def build_model(self) -> Model:
        inputs = layers.Input(shape=self.input_shape, name="input_layer")
        x = layers.Conv2D(16, (3, 3), padding="same", name="conv1_1")(inputs)
        x = layers.Activation("relu", name="relu1_1")(x)
        x = layers.BatchNormalization(name="bn1_1")(x)
        x = layers.Conv2D(16, (3, 3), padding="same", name="conv1_2")(x)
        x = layers.Activation("relu", name="relu1_2")(x)
        x = layers.BatchNormalization(name="bn1_2")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool1")(x)
        x = layers.Dropout(self.dropout_rate * 0.5, name="dropout1")(x)
        x = layers.Conv2D(32, (3, 3), padding="same", name="conv2_1")(x)
        x = layers.Activation("relu", name="relu2_1")(x)
        x = layers.BatchNormalization(name="bn2_1")(x)
        x = layers.Conv2D(32, (3, 3), padding="same", name="conv2_2")(x)
        x = layers.Activation("relu", name="relu2_2")(x)
        x = layers.BatchNormalization(name="bn2_2")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="maxpool2")(x)
        x = layers.Dropout(self.dropout_rate * 0.5, name="dropout2")(x)
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(64, activation="relu", name="dense1")(x)
        x = layers.BatchNormalization(name="bn_dense")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout_dense")(x)
        outputs = layers.Dense(2, activation="softmax", name="output")(x)
        return Model(inputs, outputs, name="LivenessNet")


class AttackNetV1(BaseArchitecture):
    def build_model(self) -> Model:
        x = layers.BatchNormalization(momentum=0.9, name="other_bn")(x)
'''

P4, P12, P12C, DEGEN = "20261106-SESSABLRUN-aaaaaaa", "20261109-SESSABLRUN-bbbbbbb", "20261111-SESSABLRUN-ccccccc", "20261107-DEGEN-ddddddd"
N_FIT = {"arm_b": 1400, "arm_a_star": 1400, "arm_b_prime": 934}
SPE = {"arm_b": 88, "arm_a_star": 88, "arm_b_prime": 59}
# i tre degeneri della fixture: epoca ripristinata, accuratezza di train all'epoca 1, AUROC del dev
DEG = {("arm_b", 1, 42): (1, 0.95, 0.9975), ("arm_b_prime", 2, 44): (2, 0.92, 0.5), ("arm_a_star", 3, 44): (1, 0.97, 0.2114)}
BOUNDARY = ("arm_b", 3, 44)          # non degenere, mai recuperato, ripristina l'epoca 3 a 0,2375
FAST = ("arm_a_star", 1, 43)         # non degenere con accuratezza di train 0,99 all'epoca 1
CRIT_LOSS = ("arm_a_star", 4, 42)    # accuratezza >= 0,75 ma val_loss >= 1,0: ramo val_loss del criterio
BOUND_HALF = ("arm_a_star", 4, 43)   # epoca ripristinata 3 con dev 0,5: non degenere per definizione
SLOW_THRESHOLD = ("arm_b_prime", 4, 42)
SLOW_EPOCH = ("arm_b", 2, 42)
# epoche a 0,5 dei degeneri nelle due run a 12 epoche
P12_HALF = {("arm_b", 1, 42): [1, 2, 3, 4, 5], ("arm_b_prime", 2, 44): [1, 2, 3, 4, 5, 6, 7], ("arm_a_star", 3, 44): [1, 2]}
P12C_HALF = {("arm_b", 1, 42): [1, 2, 3, 4, 5], ("arm_b_prime", 2, 44): [1, 2, 3], ("arm_a_star", 3, 44): [1, 2]}


def models():
    return [(a, f, s) for a in N.ARMS for f in N.FOLDS for s in N.SEEDS]


def history(k):
    """(val_loss, val_accuracy, train_accuracy) per epoca della run con pazienza 4."""
    if k in DEG:
        rest, tr1, _ = DEG[k]
        if rest == 1:
            vl, va = [7.0, 11.0, 25.0, 70.0, 58.0], [0.5] * 5
        else:
            vl, va = [25.0, 11.0, 15.0, 18.0, 13.0, 24.0], [0.5, 0.5, 0.5, 0.6, 0.5, 0.7]
        return vl, va, [tr1] + [0.98] * (len(vl) - 1)
    if k == CRIT_LOSS:
        return [5.0, 4.0, 1.2, 1.5, 1.8, 2.0], [0.5, 0.5, 0.8, 0.85, 0.9, 0.9], [0.96] * 6
    if k == BOUND_HALF:
        return [5.0, 4.0, 3.0, 3.5, 4.0, 4.5, 5.0], [0.5] * 7, [0.96] * 7
    if k == BOUNDARY:
        return [5.0, 4.0, 3.0, 3.5, 4.0, 4.5, 5.0], [0.5, 0.5, 0.2375, 0.3, 0.5, 0.4, 0.45], [0.96] * 7
    vl = [5.0, 4.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.25, 0.28]
    va = [0.5, 0.5, 0.8, 0.85, 0.9, 0.9, 0.9, 0.92, 0.93, 0.95, 0.94, 0.94]
    return vl, va, [0.99 if k == FAST else 0.96] + [0.99] * 11


def restored(vl):
    return vl.index(min(vl)) + 1


def csv_text(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def write(root: Path, rel: str, text: str):
    p = root / "research" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def make_tree(root: Path):
    """Albero research/ + src/ sintetico con le 12 sorgenti."""
    arch_sha = hashlib.sha256(ARCH_TEXT.encode()).hexdigest()
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "src" / "architectures.py").write_text(ARCH_TEXT, encoding="utf-8")
    mrows, crows, training, cmp_rows, timings = [], [], {}, [], {}
    for i, k in enumerate(models()):
        a, f, s = k
        vl, va, tr = history(k)
        rest = restored(vl)
        name = "%s_fold%d_seed%d" % k
        half = sum(1 for v in va if v == 0.5)
        rec = next((e + 1 for e in range(len(vl)) if va[e] >= 0.75 and vl[e] < 1.0), "")
        spe_s = 35.0 if k == SLOW_EPOCH else {"arm_b": 30.0, "arm_a_star": 32.0, "arm_b_prime": 20.0}[a]
        mrows.append(["sessablrun", a, f, s, N_FIT[a], SPE[a], len(vl), rest, "early stopping", repr(vl[rest - 1]),
                      repr(va[rest - 1]), int(k in DEG), rec, half, spe_s * len(vl), spe_s, P4])
        for e in range(len(vl)):
            crows.append(["sessablrun", a, f, s, N_FIT[a], SPE[a], e + 1, 0.1, repr(tr[e]), repr(vl[e]), repr(va[e]), 0.001,
                          int(e + 1 == rest), int(k in DEG), P4, len(vl)])
        training[name] = {"arm": a, "fold": f, "seed": s, "epochs_run": len(vl), "restored_epoch": rest,
                          "best_val_loss": vl[rest - 1], "n_fitting": N_FIT[a], "n_dev": 400,
                          "weights_sha256": hashlib.sha256(name.encode()).hexdigest()}
        auroc = DEG[k][2] if k in DEG else 0.99
        cmp_rows.append([a, f, s, int(k in DEG), rest, repr(auroc), "0.1234"])
        timings["train " + name] = {"monotonic_seconds": 100.0}
        timings["threshold " + name] = {"monotonic_seconds": 8.0 if k == SLOW_THRESHOLD else 4.0}
    mrows.append(["foldsrun", "folds", 1, 42, 2100, 132, 12, 5, "x", "0.1", "0.9", 0, 3, 2, 600.0, 50.0, "other"])
    write(root, N.SOURCES["models"], csv_text(["run", "arm", "fold", "seed", "n_fitting", "steps_per_epoch", "epochs_run",
                                                "restored_epoch", "restored_by", "best_val_loss", "val_acc_at_best", "degenerate",
                                                "first_recovery_epoch", "epochs_with_val_acc_0.5", "train_seconds",
                                                "seconds_per_epoch", "run_id"], mrows))
    write(root, N.SOURCES["curves"], csv_text(["run", "arm", "fold", "seed", "n_fitting", "steps_per_epoch", "epoch", "loss",
                                                "accuracy", "val_loss", "val_accuracy", "lr", "is_best", "degenerate", "run_id",
                                                "stopped_epoch"], crows))
    write(root, N.SOURCES["degenerate_aggregate"], json.dumps({
        "run_id": DEGEN, "recipe": {"bn_momentum": 0.99},
        "criteria": {"degenerate": {"restored_epochs": [1, 2], "val_acc_at_best": 0.5, "tolerance": 1e-6},
                     "recovery": {"val_accuracy_min": 0.75, "val_loss_max": 1.0}},
        "inputs": {"ablation_run": {"run_id": P4, "subjects_per_fold_dev": {"1": 2, "2": 2, "3": 2, "4": 3}}}}))
    for key, rid in (("p4_record", P4), ("p12_record", P12), ("p12c_record", P12C)):
        write(root, N.SOURCES[key], json.dumps({"run_id": rid, "status": "PASS", "dirty": False,
                                                "code_sha256": {"architecture": arch_sha}, "timings": timings}))
    write(root, N.SOURCES["p4_aggregate"], json.dumps({"run_id": P4, "training": training, "comparisons": {"not": "read"}}))
    write(root, N.SOURCES["p12_comparison"], csv_text(["arm", "fold", "seed", "in_ten", "parent_restored_epoch",
                                                        "parent_dev_auroc_at_checkpoint", "other"], cmp_rows))
    for key, halves in (("p12_curves", P12_HALF), ("p12c_curves", P12C_HALF)):
        rows = []
        for k in models():
            for e in range(1, 13):
                half = e in halves.get(k, [1])
                rows.append([k[0], k[1], k[2], int(k in DEG), e, 0.1, 0.99, 3.0 if half else 0.2, 0.5 if half else 0.95, 0.001])
        write(root, N.SOURCES[key], csv_text(["arm", "fold", "seed", "in_ten", "epoch", "loss", "accuracy", "val_loss",
                                              "val_accuracy", "lr"], rows))
    write(root, N.SOURCES["config"], json.dumps({
        "architecture": {"sha256": arch_sha},
        "arms": {a: {"total": n} for a, n in N_FIT.items()}, "dev": {"total": 400},
        "training": {"batch": 16, "device": "CPU"}, "scoring": {"batch": 16, "device": "CPU"}}))


def run_main(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = N.main(argv)
    return code, out.getvalue()


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        make_tree(self.tmp)
        self.research = self.tmp / "research"
        self.out = self.tmp / "tables"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def src(self, key) -> Path:
        return self.research / N.SOURCES[key]

    def edit_csv(self, key, fn):
        p = self.src(key)
        rows = list(csv.DictReader(io.StringIO(p.read_text())))
        header = list(rows[0])
        for r in rows:
            fn(r)
        p.write_text(csv_text(header, [[r[h] for h in header] for r in rows]))

    def edit_json(self, key, fn):
        p = self.src(key)
        d = json.loads(p.read_text())
        fn(d)
        p.write_text(json.dumps(d))

    def assertFails(self, pattern):
        with self.assertRaises(N.NoteError) as cm:
            N.build(self.research)
        self.assertRegex(str(cm.exception), pattern)


# ---------------------------------------------------------------------------------------------------------------------

class TestParsers(unittest.TestCase):
    def test_parse_csv_rows(self):
        rows = N.parse_csv(b"a,b\n1,2\n3,4\n", "x", ("a", "b"))
        self.assertEqual([r["b"] for r in rows], ["2", "4"])

    def test_parse_csv_missing_column(self):
        with self.assertRaisesRegex(N.NoteError, "missing columns c"):
            N.parse_csv(b"a,b\n1,2\n", "x", ("a", "c"))

    def test_parse_csv_short_row(self):
        with self.assertRaisesRegex(N.NoteError, "x: line 3 has fewer fields than the header"):
            N.parse_csv(b"a,b\n1,2\n3\n", "x", ("a", "b"))

    def test_parse_csv_long_row(self):
        with self.assertRaisesRegex(N.NoteError, "x: line 2 has more fields than the header"):
            N.parse_csv(b"a,b\n1,2,3\n", "x", ("a", "b"))

    def test_parse_csv_empty(self):
        with self.assertRaisesRegex(N.NoteError, "no rows"):
            N.parse_csv(b"a,b\n", "x", ("a",))

    def test_parse_json_invalid(self):
        with self.assertRaisesRegex(N.NoteError, "not valid JSON"):
            N.parse_json(b"{", "x")

    def test_parse_json_not_object(self):
        with self.assertRaisesRegex(N.NoteError, "JSON object"):
            N.parse_json(b"[1]", "x")

    def test_num_rejects_text_and_inf(self):
        self.assertEqual(N.num("0.5", "x"), 0.5)
        with self.assertRaisesRegex(N.NoteError, "not a number"):
            N.num("", "x")
        with self.assertRaisesRegex(N.NoteError, "not finite"):
            N.num("inf", "x")

    def test_by_model_duplicate_epoch(self):
        r = {"arm": "arm_b", "fold": "1", "seed": "42", "epoch": "1"}
        with self.assertRaisesRegex(N.NoteError, "duplicated epoch"):
            N.by_model([r, dict(r)], "c")

    def test_by_model_gap(self):
        rows = [{"arm": "arm_b", "fold": "1", "seed": "42", "epoch": e} for e in ("1", "3")]
        with self.assertRaisesRegex(N.NoteError, "not 1..n"):
            N.by_model(rows, "c")

    def test_half_tolerance(self):
        self.assertTrue(N.is_half(0.5000009, 1e-6))
        self.assertFalse(N.is_half(0.500002, 1e-6))
        self.assertFalse(N.is_half(0.4975, 1e-6))


class TestFormulas(unittest.TestCase):
    def test_residual_88_steps(self):
        self.assertAlmostEqual(N.residual(0.99, 88), 0.41294967113388814, places=14)
        self.assertEqual(N.f4(N.residual(0.99, 88)), "0.4129")

    def test_residual_59_steps_and_epochs(self):
        self.assertEqual(N.f4(N.residual(0.99, 59)), "0.5527")
        self.assertEqual(N.f4(N.residual(0.99, 88 * 3)), "0.0704")
        self.assertEqual(N.f4(N.residual(0.99, 59 * 7)), "0.0158")

    def test_residual_zero_steps(self):
        self.assertEqual(N.residual(0.99, 0), 1.0)

    def test_residual_invalid(self):
        for m, s in ((1.0, 5), (0.0, 5), (0.99, -1)):
            with self.assertRaises(N.NoteError):
                N.residual(m, s)

    def test_steps_per_epoch(self):
        self.assertEqual((N.steps_per_epoch(1400, 16), N.steps_per_epoch(934, 16), N.steps_per_epoch(1600, 16)), (88, 59, 100))
        with self.assertRaises(N.NoteError):
            N.steps_per_epoch(0, 16)

    def test_med_range(self):
        self.assertEqual(N.med_range([3, 1, 2, 10]), {"median": 2.5, "min": 1, "max": 10, "n": 4})
        with self.assertRaises(N.NoteError):
            N.med_range([])

    def test_word_and_thousands(self):
        self.assertEqual((N.word(2), N.word(5), N.thousands(7400)), ("two", "five", "7,400"))
        with self.assertRaises(N.NoteError):
            N.word(11)


class TestArchitecture(unittest.TestCase):
    def test_order_and_count(self):
        layers = N.parse_livenessnet(ARCH_TEXT)
        self.assertEqual(len(layers), 22)
        self.assertEqual(layers[0], (1, "Input", "input_layer"))
        self.assertEqual(layers[-1], (22, "Dense", "output"))

    def test_bn_and_dropout_positions(self):
        f = N.layer_facts(N.parse_livenessnet(ARCH_TEXT))
        self.assertEqual([(b["position"], b["name"]) for b in f["batch_normalization"]],
                         [(4, "bn1_1"), (7, "bn1_2"), (12, "bn2_1"), (15, "bn2_2"), (20, "bn_dense")])
        self.assertEqual([(d["position"], d["name"]) for d in f["dropout"]], [(9, "dropout1"), (17, "dropout2"), (21, "dropout_dense")])

    def test_first_bn_after_dropout(self):
        f = N.layer_facts(N.parse_livenessnet(ARCH_TEXT))
        self.assertEqual([(b["name"], b["preceded_by_dropout"]) for b in f["batch_normalization"] if b["first_after_dropout"]],
                         [("bn2_1", "dropout1"), ("bn_dense", "dropout2")])
        self.assertEqual([b["name"] for b in f["batch_normalization"] if b["preceded_by_dropout"]], ["bn2_1", "bn2_2", "bn_dense"])

    def test_other_class_not_read(self):
        names = [n for _, _, n in N.parse_livenessnet(ARCH_TEXT)]
        self.assertNotIn("other_bn", names)

    def test_missing_class(self):
        with self.assertRaisesRegex(N.NoteError, "LivenessNet not found"):
            N.parse_livenessnet("class Other:\n    pass\n")

    def test_momentum_refused(self):
        text = ARCH_TEXT.replace('BatchNormalization(name="bn1_1")', 'BatchNormalization(momentum=0.9, name="bn1_1")')
        with self.assertRaisesRegex(N.NoteError, "momentum"):
            N.parse_livenessnet(text)

    def test_wrong_bn_dropout_count(self):
        text = ARCH_TEXT.replace('layers.Conv2D(32, (3, 3), padding="same", name="conv2_2")', 'layers.Dropout(0.1, name="conv2_2")')
        with self.assertRaisesRegex(N.NoteError, "5 BatchNormalization and 3 Dropout layers, found 5 and 4"):
            N.layer_facts(N.parse_livenessnet(text))

    def test_total_of_22_layers(self):
        text = ARCH_TEXT.replace('        x = layers.Flatten(name="flatten")(x)\n', "")
        with self.assertRaisesRegex(N.NoteError, "22 layers, found 21"):
            N.parse_livenessnet(text)

    def test_layer_without_name(self):
        text = ARCH_TEXT.replace('layers.Flatten(name="flatten")', "layers.Flatten()")
        with self.assertRaisesRegex(N.NoteError, 'creates a layer without name='):
            N.parse_livenessnet(text)

    def test_layers_md_marks(self):
        md = N.layers_md(N.parse_livenessnet(ARCH_TEXT))
        self.assertIn("| 12 | BatchNormalization | bn2_1 | BN: recalibrated |", md)
        self.assertIn("| 21 | Dropout | dropout_dense | dropout: off in inference |", md)
        self.assertTrue(md.endswith("\n"))


class TestBuildCounts(Fixture):
    def setUp(self):
        super().setUp()
        self.doc = N.build(self.research)
        self.f = self.doc["facts"]

    def test_36_rows_sorted(self):
        rows = self.doc["rows"]
        self.assertEqual(len(rows), 36)
        self.assertEqual([(r["arm"], r["fold"], r["seed"]) for r in rows], models())

    def test_degenerate_count_and_epochs(self):
        m = self.f["models"]
        self.assertEqual((m["degenerate"], m["non_degenerate"]), (3, 33))
        self.assertEqual(m["degenerate_by_arm"], {"B": 1, "A*": 1, "B'": 1})
        self.assertEqual(m["degenerate_restored_epochs"], {"1": 2, "2": 1})
        self.assertEqual(m["degenerate_dev_accuracy_half_at_restored"], 3)

    def test_train_accuracy_epoch1(self):
        t = self.f["train_accuracy_epoch1"]
        self.assertEqual((t["degenerate"]["median"], t["degenerate"]["min"], t["degenerate"]["max"]), (0.95, 0.92, 0.97))
        self.assertEqual((t["non_degenerate"]["median"], t["non_degenerate"]["max"]), (0.96, 0.99))

    def test_residuals_per_row(self):
        r = {(x["arm"], x["fold"], x["seed"]): x for x in self.doc["rows"]}
        self.assertAlmostEqual(r[("arm_b_prime", 2, 44)]["bn_residual_restored"], 0.99 ** 118)
        self.assertAlmostEqual(r[("arm_b", 1, 42)]["bn_residual_epoch1"], 0.99 ** 88)
        self.assertAlmostEqual(r[("arm_b", 1, 43)]["bn_residual_restored"], 0.99 ** 880)
        rr = self.f["bn"]["residual_restored_degenerate"]
        self.assertAlmostEqual(rr["median"], 0.99 ** 88)
        self.assertAlmostEqual(rr["min"], 0.99 ** 118)

    def test_criterion_and_saturation_counts(self):
        m = self.f["models"]
        self.assertEqual(m["non_degenerate_at_criterion_at_checkpoint"], 30)
        ds = self.f["dev_saturation"]
        self.assertEqual((ds["degenerate_at_epoch1"], ds["non_degenerate_at_epoch1"], ds["non_degenerate_at_epoch3_or_later"]), (3, 33, 2))

    def test_criterion_val_loss_branch_and_restored_epoch_3(self):
        r = {(x["arm"], x["fold"], x["seed"]): x for x in self.doc["rows"]}
        c = r[CRIT_LOSS]
        self.assertEqual((c["dev_accuracy_restored"], c["best_val_loss"], c["at_criterion_restored"], c["first_recovery_epoch"]),
                         (0.8, 1.2, 0, None))
        b = r[BOUND_HALF]
        self.assertEqual((b["restored_epoch"], b["dev_accuracy_restored"], b["degenerate"], b["epochs_at_half"]), (3, 0.5, 0, 7))
        out = [(o["arm"], o["fold"], o["seed"]) for o in self.f["non_degenerate_outside_criterion"]]
        self.assertEqual(out, [("B", 3, 44), ("A*", 4, 42), ("A*", 4, 43)])

    def test_first_recovery_and_boundary_row(self):
        r = {(x["arm"], x["fold"], x["seed"]): x for x in self.doc["rows"]}
        b = r[BOUNDARY]
        self.assertEqual((b["restored_epoch"], b["first_recovery_epoch"], b["at_criterion_restored"], b["epochs_at_half"]), (3, None, 0, 3))
        self.assertEqual(r[("arm_b", 1, 43)]["first_recovery_epoch"], 3)

    def test_auroc_counts(self):
        a = self.f["dev_auroc_checkpoint_degenerate"]
        self.assertEqual(a["values"], [0.2114, 0.5, 0.9975])
        self.assertEqual((a["above_0.9"], a["equal_0.5"], a["below_0.5"]), (1, 1, 1))

    def test_later_runs(self):
        p, c = self.f["later_runs"]["p12"], self.f["later_runs"]["p12c"]
        self.assertEqual((p["ten_half_epochs"]["min"], p["ten_half_epochs"]["max"], p["ten_half_epochs"]["median"]), (2, 7, 5))
        self.assertEqual((p["ten_last_half_epoch"], p["ten_last_half_arms"]), (7, ["B'"]))
        self.assertAlmostEqual(p["residual_at_last_half_epoch"], 0.99 ** (59 * 7))
        self.assertEqual((c["ten_last_half_epoch"], c["ten_last_half_arms"]), (5, ["B"]))
        self.assertAlmostEqual(c["residual_at_last_half_epoch"], 0.99 ** (88 * 5))
        self.assertEqual((p["ten_with_half_epoch"], p["ten_epoch1_half"], p["others_with_half_epoch"]), (3, 3, 33))
        self.assertEqual((p["run_id"], c["run_id"]), (P12, P12C))

    def test_cost_hand_computed(self):
        c = self.f["cost"]
        self.assertEqual((c["threshold_step_seconds"]["median"], c["threshold_step_seconds"]["max"]), (4.0, 8.0))
        self.assertEqual(c["per_arm"]["B"]["frames_per_model"], 10600)
        self.assertEqual(c["per_arm"]["B'"]["frames_per_model"], 7338)
        self.assertAlmostEqual(c["per_arm"]["B"]["central_seconds_per_model"], 57.0)
        self.assertAlmostEqual(c["per_arm"]["B'"]["central_seconds_per_model"], 40.69)
        self.assertAlmostEqual(c["per_arm"]["B"]["upper_seconds_per_model"], 253.0)
        self.assertAlmostEqual(c["per_arm"]["B'"]["upper_seconds_per_model"], 148.0)
        self.assertAlmostEqual(c["total_central_minutes"], (12 * 57 + 12 * 57 + 12 * 40.69) / 60)
        self.assertAlmostEqual(c["total_upper_minutes"], 12 * (253 + 232 + 148) / 60)
        self.assertEqual((c["passes_on_fitting"], c["fitting_evaluation_passes"], c["dev_passes"]), (5, 2, 2))
        self.assertEqual(c["slowest_per_epoch_mean"], {"B": 35.0, "A*": 32.0, "B'": 20.0})
        self.assertTrue(c["derived_not_measured"])

    def test_sizes_and_runs(self):
        s = self.f["sizes"]
        self.assertEqual((s["fitting"], s["steps_per_epoch"], s["dev"], s["batch"], s["device"]),
                         ({"B": 1400, "A*": 1400, "B'": 934}, {"B": 88, "A*": 88, "B'": 59}, 400, 16, "CPU"))
        self.assertEqual(self.f["runs"]["patience_4"], P4)
        self.assertEqual(self.f["models"]["weights_sha256_listed"], 36)

    def test_sources_hashes(self):
        src = self.doc["sources"]
        self.assertEqual(len(src), 12)
        self.assertEqual(src["src/architectures.py"], hashlib.sha256(ARCH_TEXT.encode()).hexdigest())
        self.assertEqual(src[N.SOURCES["models"]], hashlib.sha256(self.src("models").read_bytes()).hexdigest())


class TestErrorPaths(Fixture):
    def test_missing_source(self):
        self.src("p12c_curves").unlink()
        self.assertFails("source missing: nuaa-constant-lr-control/comparison/curves.csv")

    def test_missing_architecture(self):
        (self.tmp / "src" / "architectures.py").unlink()
        self.assertFails("source missing: ../src/architectures.py")

    def test_degenerate_flag_with_dev_not_half(self):
        def fn(r):
            if (r["arm"], r["fold"], r["seed"], r["epoch"]) == ("arm_b", "1", "42", "1"):
                r["val_accuracy"] = "0.6"
        self.edit_csv("curves", fn)
        self.edit_csv("models", lambda r: r.update(val_acc_at_best="0.6") if (r["arm"], r["fold"], r["seed"]) == ("arm_b", "1", "42") else None)
        self.assertFails("flagged degenerate=1 but the curves give restored epoch 1 with dev accuracy 0.6")

    def test_flag_missing_on_a_degenerate_model(self):
        self.edit_csv("models", lambda r: r.update(degenerate="0") if (r["arm"], r["fold"], r["seed"]) == ("arm_b", "1", "42") else None)
        self.assertFails("flagged degenerate=0")

    def test_restored_epoch_not_minimum(self):
        self.edit_csv("models", lambda r: r.update(restored_epoch="2") if (r["arm"], r["fold"], r["seed"]) == ("arm_b", "1", "42") else None)
        self.assertFails("not the first epoch of minimum val_loss")

    def test_models_and_curves_disagree(self):
        self.edit_csv("models", lambda r: r.update(best_val_loss="7.5") if (r["arm"], r["fold"], r["seed"]) == ("arm_b", "1", "42") else None)
        self.assertFails("disagree at the restored epoch")

    def test_n_fitting_against_config(self):
        self.edit_json("config", lambda d: d["arms"]["arm_b_prime"].update(total=900))
        self.assertFails("n_fitting 934 differs from the configuration")

    def test_steps_per_epoch_mismatch(self):
        self.edit_json("config", lambda d: d["training"].update(batch=32))
        self.assertFails("steps per epoch")

    def test_architecture_hash_mismatch(self):
        self.edit_json("p12_record", lambda d: d["code_sha256"].update(architecture="0" * 64))
        self.assertFails("p12 run record: architecture SHA-256")

    def test_dirty_record(self):
        self.edit_json("p4_record", lambda d: d.update(dirty=True))
        self.assertFails("clean tree")

    def test_gpu_device_refused(self):
        self.edit_json("config", lambda d: d["scoring"].update(device="GPU"))
        self.assertFails("device must be CPU")

    def test_epochs_at_half_mismatch(self):
        self.edit_csv("models", lambda r: r.update(**{"epochs_with_val_acc_0.5": "9"}) if r["run"] == "sessablrun" and r["arm"] == "arm_b" and r["fold"] == "1" and r["seed"] == "42" else None)
        self.assertFails("epochs at 0.5 in the curves")

    def test_first_recovery_mismatch(self):
        self.edit_csv("models", lambda r: r.update(first_recovery_epoch="4") if (r["arm"], r["fold"], r["seed"]) == ("arm_b", "1", "43") else None)
        self.assertFails("first recovery epoch 3")

    def test_bad_weights_hash(self):
        self.edit_json("p4_aggregate", lambda d: d["training"]["arm_b_fold1_seed42"].update(weights_sha256="xyz"))
        self.assertFails("weights_sha256 is not a SHA-256")

    def test_training_block_disagrees(self):
        self.edit_json("p4_aggregate", lambda d: d["training"]["arm_b_fold1_seed42"].update(restored_epoch=3))
        self.assertFails("training block of the patience-4 aggregate")

    def test_comparison_restored_epoch(self):
        self.edit_csv("p12_comparison", lambda r: r.update(parent_restored_epoch="4") if (r["arm"], r["fold"], r["seed"]) == ("arm_b", "1", "42") else None)
        self.assertFails("comparison.csv has no parent row")

    def test_missing_threshold_timing(self):
        self.edit_json("p4_record", lambda d: d["timings"].pop("threshold arm_b_fold1_seed42"))
        self.assertFails("no threshold timing")

    def test_in_ten_mismatch(self):
        self.edit_csv("p12_curves", lambda r: r.update(in_ten="1") if (r["arm"], r["fold"], r["seed"]) == ("arm_b", "1", "43") else None)
        self.assertFails("in_ten of arm_b fold 1 seed 43")

    def test_per_epoch_mean_must_match_train_seconds(self):
        self.edit_csv("models", lambda r: r.update(train_seconds="1.0") if (r["arm"], r["fold"], r["seed"]) == ("arm_b", "2", "42") else None)
        self.assertFails("seconds_per_epoch 35.0 is not train_seconds / epochs_run")

    def test_short_row_names_file_and_line(self):
        p = self.src("models")
        lines = p.read_text().splitlines(keepends=True)
        lines[3] = ",".join(lines[3].split(",")[:5]) + "\n"
        p.write_text("".join(lines))
        self.assertFails("models.csv: line 4 has fewer fields than the header")

    def test_missing_model(self):
        p = self.src("models")
        lines = p.read_text().splitlines(keepends=True)
        p.write_text("".join(l for l in lines if not l.startswith("sessablrun,arm_b,4,44,")))
        self.assertFails("36 models of the patience-4 run")

    def test_degenerate_criterion_changed(self):
        self.edit_json("degenerate_aggregate", lambda d: d["criteria"]["degenerate"].update(restored_epochs=[1, 2, 3]))
        self.assertFails("degenerate criterion differs")


class TestOutputs(Fixture):
    def test_checkpoints_csv(self):
        files = N.outputs(N.build(self.research))
        rows = list(csv.DictReader(io.StringIO(files["checkpoints.csv"])))
        self.assertEqual(tuple(rows[0]), N.CHECKPOINT_COLUMNS)
        self.assertEqual(len(rows), 36)
        b = next(r for r in rows if (r["arm"], r["fold"], r["seed"]) == ("arm_b", "3", "44"))
        self.assertEqual((b["first_recovery_epoch"], b["bn_residual_restored"]), ("NA", "%.6f" % 0.99 ** 264))

    def test_deterministic_and_lf(self):
        a, b = N.outputs(N.build(self.research)), N.outputs(N.build(self.research))
        self.assertEqual(a, b)
        for name, text in a.items():
            self.assertNotIn("\r", text, name)
            self.assertTrue(text.endswith("\n"), name)

    def test_json_sorted_keys(self):
        files = N.outputs(N.build(self.research))
        for name in ("facts.json", "sources.json"):
            d = json.loads(files[name])
            self.assertEqual(files[name], json.dumps(d, indent=1, sort_keys=True) + "\n")

    def test_facts_md_lines(self):
        md = N.outputs(N.build(self.research))["facts.md"]
        self.assertIn("| degenerate (restored epoch 1 or 2 with dev accuracy 0.5 (within 1e-06)) | 3 of 36 (B 1, A* 1, B' 1) |", md)
        self.assertIn("| first BN layer after each dropout | bn2_1 after dropout1, bn_dense after dropout2 |", md)
        self.assertIn("3 of 3; 2-7 (median 5); 3 of 3; epoch 7 (B', 0.0158)", md)

    def test_note_numbers_from_facts(self):
        nums = N.note_numbers(N.build(self.research)["facts"])
        self.assertIn("dev accuracy 0.5 at epoch 1 in 3 of 3 and 33 of 33", nums)
        self.assertIn("A* fold 4 seed 42 (restored epoch 3, dev accuracy 0.8000, not degenerate)", nums)
        self.assertIn("0.9500 [0.9200-0.9700] (the three), 0.9600 (the 33)", nums)
        self.assertIn("epoch 7 (B', residual 0.0158)", nums)
        self.assertIn("residual (weight of the initial value) 0.4129 after epoch 1 (B, A\\*; B' 0.5527)", nums)
        self.assertIn("B fold 3 seed 44 (restored epoch 3, dev accuracy 0.2375, not degenerate)", nums)
        self.assertIn("A* fold 4 seed 43 (restored epoch 3, dev accuracy 0.5000, not degenerate)", nums)
        self.assertIn("(10,600 frame passes for B and A\\*, 7,338 for B')", nums)
        self.assertIn("five recalibration and two evaluation passes over the fitting set, two over the dev", nums)
        self.assertIn("Small dev (2-3 subjects)", nums)
        self.assertIn("the one that already order the dev (AUROC 0.9975, 0.5000)", nums)
        self.assertIn("one of the three have dev AUROC 0.5000", nums)
        self.assertIn("The one tied at AUROC 0.5000", nums)
        self.assertIn("hypotheses 4 and 7", nums)
        self.assertIn("the three, saturated at epoch 1 or 2; the 33 as controls", nums)
        self.assertIn("the five BatchNormalization layers", nums)
        self.assertIn("dev accuracy >= 0.75, val_loss < 1.0", nums)
        for s in nums:
            self.assertNotIn("\n", s)


class TestCommands(Fixture):
    def build_cmd(self):
        return run_main(["build", "--research", str(self.research), "--out", str(self.out)])

    def check_cmd(self, *extra):
        return run_main(["check", "--research", str(self.research), "--out", str(self.out), *extra])

    def note(self, drop=None):
        nums = N.note_numbers(N.build(self.research)["facts"])
        text = "# Note\n\n" + "\n".join("Fact %d: %s." % (i, s) for i, s in enumerate(nums) if s != drop) + "\n"
        p = self.tmp / "NOTE.md"
        p.write_text(text)
        return p

    def test_build_writes_five_files(self):
        code, out = self.build_cmd()
        self.assertEqual(code, 0, out)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), sorted(N.OUTPUTS))
        self.assertIn("BUILD: 36 checkpoints, 3 degenerate, 12 sources", out)

    def test_check_passes(self):
        self.build_cmd()
        code, out = self.check_cmd()
        self.assertEqual(code, 0, out)
        self.assertIn("CHECK: 5 files, 0 problems", out)

    def test_check_detects_edited_table(self):
        self.build_cmd()
        p = self.out / "facts.md"
        p.write_text(p.read_text().replace("3 of 36", "4 of 36"))
        code, out = self.check_cmd()
        self.assertEqual(code, 1)
        self.assertIn("PROBLEM: facts.md: differs from the rebuilt file", out)

    def test_check_detects_missing_table(self):
        self.build_cmd()
        (self.out / "layers.md").unlink()
        code, out = self.check_cmd()
        self.assertEqual(code, 1)
        self.assertIn("PROBLEM: layers.md: missing", out)

    def test_check_detects_source_hash_change(self):
        self.build_cmd()
        p = self.src("p12_comparison")
        p.write_text(p.read_text() + "\n")
        code, out = self.check_cmd()
        self.assertEqual(code, 1)
        self.assertIn("nuaa-patience-verification/comparison/comparison.csv: SHA-256 differs from the one recorded", out)

    def test_check_detects_missing_source(self):
        self.build_cmd()
        self.src("config").unlink()
        code, out = self.check_cmd()
        self.assertEqual(code, 1)
        self.assertIn("nuaa_session_ablation_run_constant_lr.json: source missing", out)

    def test_check_without_build(self):
        code, out = self.check_cmd()
        self.assertEqual(code, 1)
        self.assertIn("ERROR:", out)
        self.assertIn("run build first", out)

    def test_check_note_passes(self):
        self.build_cmd()
        code, out = self.check_cmd("--note", str(self.note()))
        self.assertEqual(code, 0, out)
        self.assertIn("CHECK: 5 files + note, 0 problems", out)

    def test_check_note_missing_number(self):
        self.build_cmd()
        f = N.build(self.research)["facts"]
        drop = "about %.0f minutes for 36 models, upper estimate about %.0f minutes" % (f["cost"]["total_central_minutes"],
                                                                                    f["cost"]["total_upper_minutes"])
        code, out = self.check_cmd("--note", str(self.note(drop=drop)))
        self.assertEqual(code, 1)
        self.assertIn("NOTE.md: %r not found" % drop, out)

    def test_check_note_every_fragment_required(self):
        """Ogni frammento della lista, tolto dalla nota, fa fallire il controllo."""
        self.build_cmd()
        for drop in N.note_numbers(N.build(self.research)["facts"]):
            with self.subTest(drop=drop):
                code, out = self.check_cmd("--note", str(self.note(drop=drop)))
                self.assertEqual(code, 1)
                self.assertIn("not found", out)

    def test_check_note_ignores_line_breaks(self):
        self.build_cmd()
        p = self.note()
        p.write_text(p.read_text().replace("3 of 36 models", "3 of\n36   models"))
        code, out = self.check_cmd("--note", str(p))
        self.assertEqual(code, 0, out)

    def test_error_without_traceback(self):
        self.src("curves").write_text("run,arm\nsessablrun,arm_b\n")
        code, out = self.build_cmd()
        self.assertEqual(code, 1)
        self.assertIn("ERROR: nuaa-degenerate-trainings/tables/curves.csv: missing columns", out)
        self.assertNotIn("Traceback", out)
        self.assertTrue(out.rstrip().endswith("RESULT: FAIL"))

    def test_help(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as cm:
            N.main(["--help"])
        self.assertEqual(cm.exception.code, 0)
        self.assertIn("build", out.getvalue())


class TestRealPublicFiles(unittest.TestCase):
    """Da capo a fondo sulle sorgenti pubbliche reali del repository, se presenti."""

    def setUp(self):
        self.research = REPO / "research"
        self.pkg = self.research / "nuaa-saturation-control"
        if not all((self.research / rel).is_file() for rel in N.SOURCES.values()):
            self.skipTest("public sources not present in %s" % self.research)

    def test_check_committed_tables_and_note(self):
        code, out = run_main(["check", "--research", str(self.research), "--out", str(self.pkg / "tables"),
                              "--note", str(self.pkg / "NOTE.md")])
        self.assertEqual(code, 0, out)

    def test_real_facts(self):
        f = N.build(self.research)["facts"]
        self.assertEqual((f["models"]["total"], f["models"]["degenerate"]), (36, 10))
        self.assertEqual(f["models"]["degenerate_restored_epochs"], {"1": 8, "2": 2})
        self.assertEqual(f["models"]["degenerate_dev_accuracy_half_at_restored"], 10)
        self.assertEqual(N.f4(f["bn"]["residual_after_epoch"]["B"][0]), "0.4129")

    def test_no_subject_identifier_in_outputs(self):
        files = N.outputs(N.build(self.research))
        files["NOTE.md"] = (self.pkg / "NOTE.md").read_text(encoding="utf-8")
        for name, text in files.items():
            self.assertIsNone(re.search(r"(?<![0-9A-Za-z.])00\d\d(?![0-9A-Za-z])", text), name)


if __name__ == "__main__":
    unittest.main()
