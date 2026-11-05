"""
Test di scripts/nuaa_session_ablation.py (conteggi per la proposta «training con e senza la sessione 03»), offline e
deterministici.

Fixture sintetiche con identificativi fuori da quelli NUAA (1101–1116): liste ufficiali, configurazione e run dello
schema dei fold prodotti con il codice approvato (scripts/nuaa_subject_folds.py, attraverso la fixture di
scripts/check_nuaa_subject_folds.py) su una struttura che ricalca quella reale ma con conteggi diversi (3 soggetti A,
6 B, 6 C, 1 D; un soggetto B senza bona fide nella sessione 02, uno senza bona fide nella 01, un C senza attacchi).
Risposte calcolate a mano (commenti accanto ai valori attesi): fold, soggetti S12, disponibilità per cella e fold,
n del braccio B, ripartizione del braccio A* (bilanciata e sbilanciata), dev comune e alternativo, campionamento
`spread`, disgiunzione dei ruoli, budget, controllo di rango della riservatezza su esempi piccoli. Percorsi d'errore:
SHA-256 della configurazione dello schema, dei manifest M2, dei SHA256SUMS e dei manifest dello schema, run dello
schema non PASS, manifest dello schema manomesso o incompleto, JSON pubblico dello schema in disaccordo con il
ricalcolo, configurazione senza campi, con regole o valori diversi da quelli delle regole, budget oltre il tetto,
collegamento simbolico, run senza --i-am-authorized, cartella estranea o protetta, ingresso dentro il run → FAIL senza
uscite; nessun identificativo né conteggio trattenuto nel pubblico; nessun PASS vecchio dopo un FAIL. Un test usa gli
ingressi veri (sola lettura) e si salta solo se mancano (MISSING_INPUT). Alla fine si controlla che results/, export/
e delivery/ del repository non siano cambiati.

  .venv/bin/python scripts/check_nuaa_session_ablation.py        # uscita 1 se un test fallisce

Variabile facoltativa SESSABL_INPUT_ROOT: radice degli ingressi veri (predefinita: la radice del repository; la usa il
controllo delle mutazioni, che esegue questa suite su una copia del codice).
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_nuaa_subject_folds as CF  # noqa: E402
import nuaa_session_ablation as SA  # noqa: E402
import nuaa_subject_folds as NF  # noqa: E402

REAL_ROOT = Path(os.environ.get("SESSABL_INPUT_ROOT") or ROOT)
STATE = {"commit": "abcdef0", "dirty": False}
RUN_ID = "20261104-SESSABL-abcdef0"
FOLDS_RUN = "20261103-FOLDS-abcdef0"
MISSING_INPUT = {"real inputs": "the restricted M2 manifests or the fold schema run are not present"}
CELLS = SA.CELLS


def skip_missing(test: unittest.TestCase, what: str):
    """Unico modo di saltare un test di questa suite: un ingresso mancante dichiarato in MISSING_INPUT."""
    test.skipTest(f"missing input: {MISSING_INPUT[what]}")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ============================================================================ fixture

# soggetto -> frame per sessione 01 (bf, att), 02 (bf, att), 03 (bf, att); identificativi e conteggi finti
SUBJECTS = {
    "1101": ((40, 30), (42, 30), (90, 50)), "1102": ((36, 30), (38, 30), (55, 45)), "1103": ((44, 30), (40, 30), (85, 52)),
    "1104": ((38, 30), (36, 30), (0, 48)), "1105": ((30, 28), (0, 30), (0, 47)), "1106": ((34, 29), (37, 30), (0, 49)),
    "1107": ((41, 30), (39, 30), (0, 51)), "1108": ((0, 27), (35, 30), (0, 46)), "1109": ((33, 26), (37, 30), (0, 50)),
    "1110": ((0, 0), (0, 0), (20, 40)), "1111": ((0, 0), (0, 0), (54, 44)), "1112": ((0, 0), (0, 0), (57, 41)),
    "1113": ((0, 0), (0, 0), (59, 0)), "1114": ((0, 0), (0, 0), (61, 43)), "1115": ((0, 0), (0, 0), (25, 42)),
    "1116": ((0, 0), (0, 0), (0, 53)),
}
# a mano, regola bilanciata dello schema: A 1101→1, 1102→2, 1103→3; B 1104→4, 1105→1, 1106→2, 1107→3, 1108→4, 1109→1;
# C (meno bona fide 03 a parità) 1110→4, 1111→2, 1112→3, 1113→1, 1114→4, 1115→2; D 1116→3
FOLDS = {1: ["1101", "1105", "1109", "1113"], 2: ["1102", "1106", "1111", "1115"], 3: ["1103", "1107", "1112", "1116"],
         4: ["1104", "1108", "1110", "1114"]}
# soggetti S12 (fitting con sessioni 01-02) per fold tenuto fuori: 1 → fold 3, 4; 2 → fold 4, 1; 3 → fold 1, 2; 4 → fold 2, 3
S12 = {1: {"1103", "1107", "1104", "1108"}, 2: {"1104", "1108", "1101", "1105", "1109"},
       3: {"1101", "1105", "1109", "1102", "1106"}, 4: {"1102", "1106", "1103", "1107"}}
# disponibilità S12 per cella (01/bf, 01/att, 02/bf, 02/att, 03/bf, 03/att), a mano dalla tabella SUBJECTS
S12_AVAIL = {1: [123, 117, 150, 120, 85, 197],       # 44+41+38+0, 30+30+30+27, 40+39+36+35, 4x30, 1103 solo, 52+51+48+46
             2: [141, 141, 150, 150, 90, 241],       # 38+0+40+30+33, 30+27+30+28+26, 36+35+42+0+37, 5x30, 1101 solo, ...
             3: [173, 143, 154, 150, 145, 241],      # 40+30+33+36+34, ..., 42+0+37+38+37, 5x30, 90+55, 50+47+50+45+49
             4: [155, 119, 154, 120, 140, 197]}      # 36+34+44+41, 30+29+30+30, 38+37+40+39, 4x30, 55+85, 45+49+52+51
N_ARM_B = 110            # minimo 117 (fold 1, 01/attack), passo 10
ALLOC_A = {"01/bona_fide": 74, "01/attack": 73, "02/bona_fide": 73, "02/attack": 74, "03/bona_fide": 73, "03/attack": 73}
DEV_N = 30               # minimo 38 (dev del fold 3 = fold 4, 01/bona_fide di 1104 solo), passo 10
ALLOC_DEV = {c: 20 for c in CELLS}     # 120 / 6 senza resto
TEST_FRAMES = [652, 578, 623, 484]     # 282+135+176+59, 234+179+98+67, 281+191+98+53, 182+138+60+104
NF_FIT, NF_DEV = 100, 30               # conteggi dello schema sintetico (fitting minimo 117, dev minimo 38)
ORDER = ["01/bona_fide", "02/attack", "03/bona_fide", "01/attack", "02/bona_fide", "03/attack"]


def cell_count(subjects, cell) -> int:
    i = CELLS.index(cell)
    return sum(SUBJECTS[s][i // 2][i % 2] for s in subjects)


class Fixture:
    """Repository finto con liste M2 e schema dei fold sintetici (codice approvato) e configurazione coerente."""

    def __init__(self, subjects=SUBJECTS, design=None, step=10):
        self.ff = CF.Fixture(subjects, fit=NF_FIT, dev=NF_DEV)
        self.tmp, self.root, self.mdir = self.ff.tmp, self.ff.root, self.ff.mdir
        out, err = io.StringIO(), io.StringIO()
        with CF.mock_root(self.root), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = NF.main(self.ff.args("run", date="20261103"))
        if code:
            raise RuntimeError("synthetic fold schema failed: " + err.getvalue())
        self.frun = self.root / "results" / "folds" / FOLDS_RUN
        (self.root / "results" / "sessionablation").mkdir(parents=True)
        for f in SA.CODE_FILES:
            (self.root / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / f, self.root / f)
        self.cfg_path = self.root / "delivery" / "sessionablation" / "nuaa_session_ablation.json"
        self.cfg_path.parent.mkdir(parents=True)
        self.cfg = self.make_config(design or {}, step)
        self.write_config()

    def make_config(self, design, step) -> dict:
        st = json.loads((self.frun / "status.json").read_text())
        d = {"s12": SA.S12_RULE, "dev_subjects": SA.DEV_RULE, "rounding_step": step, "rounding_rule": SA.ROUNDING_RULE,
             "allocation_rule": SA.ALLOCATION_RULE, "remainder_order": list(ORDER), "sampling": SA.SAMPLING_RULE,
             "nesting": SA.NESTING_RULE, "arm_b_prime": {"cells": SA.CELLS_12, "allocation": {c: ALLOC_A[c] for c in SA.CELLS_12}},
             "arm_b": {"cells": SA.CELLS_12, "per_cell": N_ARM_B},
             "arm_a_star": {"cells": CELLS, "total": 4 * N_ARM_B, "variant": "balanced", "allocation": dict(ALLOC_A)},
             "dev": {"cells": SA.CELLS_12, "per_cell": DEV_N},
             "dev_mix": {"cells": CELLS, "total": 4 * DEV_N, "variant": "balanced", "allocation": dict(ALLOC_DEV)}}
        d.update(design)
        return {
            "schema": SA.CONFIG_SCHEMA, "status": "test", "task": "test",
            "inputs": {"folds_config": {"path": "delivery/folds/nuaa_subject_folds.json", "sha256": sha(self.ff.cfg_path.read_bytes())},
                       "fold_schema": {"run_id": FOLDS_RUN, "public_sha256sums": st["public_sha256sums"],
                                       "restricted_sha256sums": st["restricted_sha256sums"]}},
            "expected": {"k": 4, "n_subjects": 16, "n_images": len(self.ff.train) + len(self.ff.test), "test_frames": list(TEST_FRAMES)},
            "design": d, "arms": {"arm_b": "test"}, "train_seeds": [42, 43, 44], "training": {"x": 1}, "threshold": {"x": 1},
            "metrics": {"x": 1}, "declarations": {"d": 1},
            "budget": {"reference_run": "ref", "reference_run_record_sha256": "0" * 64, "reference_fitting_frames": 2100,
                       "reference_dev_frames": 600, "reference_scored_frames_per_seed": 2337,
                       "reference_train_seconds": [600.0, 900.0], "reference_score_seconds": [10.0, 20.0],
                       "reference_threshold_seconds": [3.0, 6.0], "wall_time_cap_minutes": 1000, "on_cap_exceeded": "stop"},
        }

    def write_config(self):
        self.cfg_path.write_text(json.dumps(self.cfg, indent=1) + "\n", encoding="utf-8")

    def tamper(self, rel: str, data: bytes, reanchor=True):
        """Riscrive un file del run dello schema; con reanchor aggiorna SHA256SUMS, status.json e i pin della
        configurazione, così il guasto raggiunge il controllo successivo (non quello del SHA)."""
        p = self.frun / rel
        p.write_bytes(data)
        if not reanchor:
            return
        sub = rel.split("/")[0]
        files = {n: (self.frun / sub / n).read_text() for n in sorted(os.listdir(self.frun / sub)) if n != "SHA256SUMS"
                 and n != "public_SHA256SUMS.sha256"}
        if sub == "restricted":
            files["public_SHA256SUMS.sha256"] = (self.frun / sub / "public_SHA256SUMS.sha256").read_text()
        sums = NF.sums_text(files)
        (self.frun / sub / "SHA256SUMS").write_text(sums)
        st = json.loads((self.frun / "status.json").read_text())
        key = "public_sha256sums" if sub == "public" else "restricted_sha256sums"
        st[key] = sha(sums.encode())
        (self.frun / "status.json").write_text(json.dumps(st, indent=1) + "\n")
        self.cfg["inputs"]["fold_schema"][key] = st[key]
        self.write_config()

    def args(self, cmd, out_root=None, date="20261104", authorized=True, manifests="manifests", folds_root="results/folds"):
        a = [cmd, "--config", str(self.cfg_path.relative_to(self.root)), "--manifests", str(manifests), "--folds-root", str(folds_root),
             "--out-root", str(out_root or (self.root / "results" / "sessionablation")), "--date", date]
        return a + (["--i-am-authorized"] if authorized and cmd == "run" else [])

    def run(self, cmd, **kw) -> tuple:
        out, err = io.StringIO(), io.StringIO()
        with mock_root(self.root), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = SA.main(self.args(cmd, **kw))
        return code, out.getvalue(), err.getvalue()

    def compute(self):
        a = argparse.Namespace(manifests="manifests", folds_root="results/folds")
        cfg, _ = SA.load_config(self.cfg_path)
        return SA.compute(cfg, self.root, a)

    def close(self):
        self.ff.close()


@contextlib.contextmanager
def mock_root(root: Path):
    old_root, old_state = SA.ROOT, SA.REPO_STATE
    SA.ROOT, SA.REPO_STATE = root, lambda r: dict(STATE)
    try:
        yield
    finally:
        SA.ROOT, SA.REPO_STATE = old_root, old_state


def snapshot(*dirs) -> dict:
    out = {}
    for d in dirs:
        for p in sorted(Path(d).rglob("*")):
            if p.is_file():
                out[str(p)] = (p.stat().st_size, p.stat().st_mtime_ns)
    return out


# ============================================================================ test

class TestDesign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = Fixture()
        cls.inp, cls.des, cls.base, cls.dec, cls.leaks, cls.bud = cls.fx.compute()

    @classmethod
    def tearDownClass(cls):
        cls.fx.close()

    def test_fold_members_by_hand(self):
        self.assertEqual({i: sorted(v) for i, v in self.inp["members"].items()}, {i: sorted(v) for i, v in FOLDS.items()})

    def test_roles_and_s12_by_hand(self):
        for i, f in self.des["folds"].items():
            self.assertEqual(f["dev_fold"], i % 4 + 1)
            self.assertEqual(f["s12"], S12[i], i)
            self.assertEqual(f["s03only"], f["fitting"] - S12[i])
            self.assertEqual(f["dev"], set(FOLDS[i % 4 + 1]))
        self.assertEqual(self.des["folds"][1]["s03only"], {"1112", "1116", "1110", "1114"})
        self.assertEqual(self.des["folds"][1]["fitting_folds"], [3, 4])

    def test_availability_by_hand(self):
        for i, f in self.des["folds"].items():
            self.assertEqual([len(f["cells_s12"][c]) for c in CELLS], S12_AVAIL[i], i)
            self.assertEqual([len(f["cells_dev"][c]) for c in CELLS], [cell_count(FOLDS[i % 4 + 1], c) for c in CELLS])

    def test_arm_b_n_by_hand(self):
        self.assertEqual((self.des["min_b"], self.des["n"]), (117, N_ARM_B))

    def test_arm_a_star_balanced_by_hand(self):
        self.assertEqual((self.des["alloc_a"], self.des["variant_a"]), (ALLOC_A, "balanced"))
        self.assertEqual(sum(self.des["alloc_a"].values()), 4 * N_ARM_B)
        self.assertEqual(self.des["caps_a"]["03/bona_fide"], 85)          # min(85, 90, 145, 140)

    def test_dev_and_dev_mix_by_hand(self):
        self.assertEqual((self.des["min_dev"], self.des["dev_n"]), (38, DEV_N))
        self.assertEqual((self.des["alloc_m"], self.des["variant_m"]), (ALLOC_DEV, "balanced"))

    def test_caps_by_hand(self):
        # minimo sui fold della disponibilità per cella: S12 (S12_AVAIL) e dev (fold di dev k mod 4 + 1, tutti i soggetti)
        self.assertEqual([self.des["caps_a"][c] for c in CELLS], [123, 117, 150, 120, 85, 197])
        # dev dei fold 1..4 = fold 2, 3, 4, 1: 01/bf 70, 85, 38, 103; 01/att 59, 60, 57, 84; 02/bf 75, 79, 71, 79;
        # 02/att 60, 60, 60, 90; 03/bf 134, 142, 81, 149; 03/att 180, 197, 177, 147
        self.assertEqual([self.des["caps_m"][c] for c in CELLS], [38, 57, 71, 60, 81, 147])

    def test_sample_shortfall_fails(self):
        rows = self.des["folds"][1]["cells_s12"]["03/bona_fide"]
        self.assertEqual(len(SA.sample(rows, 85, "x")), 85)
        with self.assertRaisesRegex(SA.Failure, "x: 85 frames available, 86 required"):
            SA.sample(rows, 86, "x")

    def test_allocate_remainder_order_keeps_classes_balanced(self):
        a = SA.allocate(1400, CELLS, ORDER)
        self.assertEqual([a[c] for c in CELLS], [234, 233, 233, 234, 233, 233])              # l'esempio del coordinatore
        self.assertEqual(sum(a[c] for c in CELLS if "bona_fide" in c), sum(a[c] for c in CELLS if "attack" in c))
        a = SA.allocate(404, CELLS, ORDER)                                                   # 67 x 6 + 2
        self.assertEqual([a[c] for c in CELLS], [68, 67, 67, 68, 67, 67])

    def test_allocate_capped_by_hand(self):
        caps = {c: 1000 for c in CELLS}
        caps["03/bona_fide"] = 58
        a, v = SA.allocate_capped(440, CELLS, ORDER, caps)
        # 03/bf = 58; 382 sulle altre 5: 76 e resto 2 a 01/bf e 02/att
        self.assertEqual((a, v), ({"01/bona_fide": 77, "01/attack": 76, "02/bona_fide": 76, "02/attack": 77, "03/bona_fide": 58,
                                   "03/attack": 76}, "unbalanced"))
        caps["01/attack"] = 70           # dopo il primo giro anche 01/attack è corta: 70; 312 sulle altre 4 = 78
        a, v = SA.allocate_capped(440, CELLS, ORDER, caps)
        self.assertEqual(a, {"01/bona_fide": 78, "01/attack": 70, "02/bona_fide": 78, "02/attack": 78, "03/bona_fide": 58, "03/attack": 78})
        with self.assertRaisesRegex(SA.Failure, "every cell is short"):
            SA.allocate_capped(440, CELLS, ORDER, {c: 10 for c in CELLS})

    def test_nested_spread_by_hand(self):
        # 20 nomi a00..a19: B prende 10 (passo 2: a00, a02, ..., a18); A* ne prende 4 SUI 10 scelti (passo 2,5: indici
        # 0, 2, 5, 7 dei 10 → a00, a04, a10, a14), non sui 20 disponibili (che darebbero a00, a05, a10, a15)
        rows = [{"key": f"real/a{j:02d}.jpg"} for j in range(20)]
        b = SA.sample(rows, 10, "x")
        self.assertEqual([r["key"][5:8] for r in b], [f"a{j:02d}" for j in range(0, 20, 2)])
        self.assertEqual([r["key"][5:8] for r in SA.sample(b, 4, "x")], ["a00", "a04", "a10", "a14"])

    def test_arms_nested_in_b(self):
        for i, f in self.des["folds"].items():
            m = f["manifests"]
            keys = {kind: {r["key"] for r in m[kind]} for kind in SA.MANIFEST_KINDS}
            a12 = {r["key"] for r in m["armAstar_fitting"] if r["session"] != "03"}
            self.assertTrue(a12 <= keys["armB_fitting"], i)
            self.assertEqual(keys["armBprime_fitting"], a12, i)                 # B′ = celle 01-02 di A*
            self.assertEqual(len(m["armBprime_fitting"]), 74 + 73 + 73 + 74)
            for c in SA.CELLS_12:
                bnames = sorted(r["key"].split("/", 1)[1] for r in m["armB_fitting"] if f"{r['session']}/{SA.CLASS_NAME[r['label']]}" == c)
                got = sorted(r["key"].split("/", 1)[1] for r in m["armAstar_fitting"] if f"{r['session']}/{SA.CLASS_NAME[r['label']]}" == c)
                self.assertEqual(got, NF.spread(bnames, ALLOC_A[c]), (i, c))
        self.assertEqual(self.des["alloc_bp"], {c: ALLOC_A[c] for c in SA.CELLS_12})

    def test_sampling_spread_and_disjoint(self):
        for i, f in self.des["folds"].items():
            m = f["manifests"]
            self.assertEqual(len(m["armB_fitting"]), 4 * N_ARM_B)
            self.assertEqual(len(m["armAstar_fitting"]), 4 * N_ARM_B)
            self.assertEqual(len(m["dev"]), 4 * DEV_N)
            self.assertEqual(len(m["devmix"]), 4 * DEV_N)
            for kind, src, alloc in (("armB_fitting", "cells_s12", {c: N_ARM_B for c in SA.CELLS_12}),
                                     ("armAstar_fitting", "cells_s12", {c: ALLOC_A[c] for c in SA.CELLS_03}),
                                     ("dev", "cells_dev", {c: DEV_N for c in SA.CELLS_12}), ("devmix", "cells_dev", ALLOC_DEV)):
                for c in CELLS:
                    got = sorted(r["key"].split("/", 1)[1] for r in m[kind] if f"{r['session']}/{SA.CLASS_NAME[r['label']]}" == c)
                    names = [r["key"].split("/", 1)[1] for r in f[src][c]]
                    if c in alloc or kind != "armAstar_fitting":
                        self.assertEqual(got, NF.spread(names, alloc[c]) if c in alloc else [], (i, kind, c))
            self.assertTrue({r["subject"] for r in m["armAstar_fitting"]} <= S12[i])
            self.assertFalse({r["subject"] for r in m["armB_fitting"]} & {r["subject"] for r in m["dev"]})
            self.assertFalse({r["subject"] for r in m["dev"] + m["armB_fitting"]} & set(FOLDS[i]))
            self.assertEqual(len(f["test_rows"]), TEST_FRAMES[i - 1])

    def test_span_by_hand(self):
        sp = SA.Span()
        self.assertTrue(sp.add({0: 1, 1: 1}))
        self.assertTrue(sp.add({1: 1, 2: 1}))
        self.assertFalse(sp.add({0: 2, 1: 4, 2: 2}))               # 2 (x0+x1) + 2 (x1+x2)
        self.assertTrue(sp.add({0: 1, 2: 1}))                     # indipendente sui razionali (dipendente modulo 2)
        self.assertEqual(sp.rank(), 3)
        self.assertEqual(sp.reduce({1: 5}), {})

    def test_disclosure_by_hand(self):
        atoms = {("s1", "c1"): 0, ("s2", "c1"): 1, ("s1", "c2"): 2, ("s2", "c2"): 3}
        d = SA.Disclosure(atoms)
        d.add(d.vector(["s1", "s2"], ["c1"]))
        d.add(d.vector(["s1", "s2"], ["c2"]))
        self.assertEqual(d.leaks(), {"s1": 0, "s2": 0})
        self.assertEqual(d.would_reveal(d.vector(["s1", "s2"], ["c1", "c2"])), [])          # dipendente
        # il totale di s1 rivela anche quello di s2 (somma delle due celle meno s1)
        self.assertEqual(d.would_reveal(d.vector(["s1"], ["c1", "c2"])), ["s1", "s2"])
        atoms3 = {("s1", "c"): 0, ("s2", "c"): 1, ("s3", "c"): 2}
        d3 = SA.Disclosure(atoms3)
        d3.add(d3.vector(["s1", "s2", "s3"], ["c"]))
        self.assertEqual(d3.would_reveal(d3.vector(["s1", "s2"], ["c"])), ["s3"])            # per differenza
        d3.add(d3.vector(["s1", "s2"], ["c"]))
        self.assertEqual(d3.leaks(), {"s1": 0, "s2": 0, "s3": 1})

    def test_decisions_by_hand(self):
        dec = self.dec
        self.assertEqual(dec["fold1.s12.03/bona_fide"]["status"], SA.RESTRICTED_NOTE["one"])       # 1103 solo
        self.assertEqual(dec["fold2.s12.03/bona_fide"]["status"], SA.RESTRICTED_NOTE["one"])       # 1101 solo
        self.assertEqual(dec["fold1.s12.all"]["status"], SA.RESTRICTED_NOTE["diff"])               # totale meno le celle note
        self.assertEqual(dec["fold3.s12.03/bona_fide"], {**dec["fold3.s12.03/bona_fide"], "status": "published", "value": 145})
        # dev del fold 3 = fold 4: 01/bona_fide di 1104 solo, già stampato dallo schema (disponibilità del dev)
        self.assertEqual(dec["fold3.dev.01/bona_fide"]["status"], SA.RESTRICTED_NOTE["schema"])
        self.assertEqual(dec["fold4.test.all"]["status"], "published")
        self.assertEqual(dec["fold4.test.all"]["value"], TEST_FRAMES[3])
        self.assertEqual(SA.disclosure_problems(self.des, self.base, dec), [])
        # lo schema stampa già una cella di un solo soggetto: 01/bf del fold 4 (1104; 1108 non ne ha); il 02/bf del fold 1
        # ha due soggetti (1101, 1109; 1105 non ne ha)
        self.assertEqual(sorted(s for s, v in self.leaks.items() if v), ["1104"])

    def test_disclosure_problems_catch_a_bad_publication(self):
        dec = json.loads(json.dumps({k: {**v, "vector": [sorted(v["vector"][0]), v["vector"][1]]} for k, v in self.dec.items()}))
        for k in dec:
            dec[k]["vector"] = (set(dec[k]["vector"][0]), dec[k]["vector"][1])
        dec["fold1.s12.all"]["status"] = "published"
        self.assertIn("derivable by differencing", "; ".join(SA.disclosure_problems(self.des, self.base, dec)))
        dec["fold1.s12.all"]["status"] = SA.RESTRICTED_NOTE["diff"]
        dec["fold1.s12.03/bona_fide"]["status"] = "published"
        self.assertIn("a published count of one subject", "; ".join(SA.disclosure_problems(self.des, self.base, dec)))

    def test_budget_by_hand(self):
        b = self.bud
        self.assertEqual(b["trainings"], 24)
        # rapporto (440 + 120) / 2700; media 750 s, massimo 900 s; 12 training per braccio
        r = 560 / 2700
        self.assertEqual(b["arms"]["arm_b"]["ratio_to_reference"], round(r, 4))
        self.assertEqual(b["arms"]["arm_b"]["training_minutes"], [round(750 * r * 12 / 60, 1), round(900 * r * 12 / 60, 1)])
        self.assertEqual(b["arms"]["arm_b"]["seconds_per_training"], [round(600 * r, 1), round(900 * r, 1)])
        self.assertEqual((b["test_frames_scored"], b["dev_frames_scored"]), (2 * 3 * 2337, 2 * 3 * 4 * 120))
        # punteggio: 30 s per 2337 x 2 / 4 frame; massimo 20 s per 2337 / 4; soglia 9 s per 1200 frame, massimo 6 / 600
        mid = 14022 * 30 / (2337 * 2 / 4) + 2880 * 9 / 1200
        hi = 14022 * 20 * 4 / 2337 + 2880 * 6 / 600
        self.assertEqual(b["scoring_minutes"], [round(mid / 60, 1), round(hi / 60, 1)])
        # B′ a parte: (294 + 120) / 2700, 12 training; punteggio di un braccio (metà del totale); non sommato al totale
        o, rp = b["optional"]["arm_b_prime"], 414 / 2700
        self.assertEqual((o["fitting_frames"], o["trainings"]), (294, 12))
        self.assertEqual(o["training_minutes"], [round(750 * rp * 12 / 60, 1), round(900 * rp * 12 / 60, 1)])
        self.assertEqual(o["scoring_minutes"], [round(mid / 2 / 60, 1), round(hi / 2 / 60, 1)])
        base_hi = (900 * r * 12 * 2 + hi) / 60
        self.assertEqual(b["total_minutes"][1], round(base_hi, 1))


class TestUnbalanced(unittest.TestCase):
    def test_unbalanced_variant_by_hand(self):
        subjects = dict(SUBJECTS)
        subjects["1103"] = ((44, 30), (40, 30), (58, 52))      # 03/bf di S12 nel fold 1 = 58 < 73
        alloc = {"01/bona_fide": 77, "01/attack": 76, "02/bona_fide": 76, "02/attack": 77, "03/bona_fide": 58, "03/attack": 76}
        fx = Fixture(subjects, design={"arm_a_star": {"cells": CELLS, "total": 440, "variant": "unbalanced", "allocation": alloc},
                                       "arm_b_prime": {"cells": SA.CELLS_12, "allocation": {c: alloc[c] for c in SA.CELLS_12}}})
        self.addCleanup(fx.close)
        fx.cfg["expected"]["test_frames"][2] = 623 - 27
        fx.write_config()
        inp, des, *_ = fx.compute()
        self.assertEqual((des["alloc_a"], des["variant_a"]), (alloc, "unbalanced"))
        self.assertEqual(len(des["folds"][1]["manifests"]["armAstar_fitting"]), 440)
        fx.cfg["design"]["arm_a_star"]["variant"] = "balanced"
        fx.write_config()
        code, out, err = fx.run("check-inputs")
        self.assertEqual(code, 1)
        self.assertIn("allocation or variant differ from the rule (unbalanced", err)


class TestInputs(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture()
        self.addCleanup(self.fx.close)

    def check_fails(self, pattern: str):
        code, out, err = self.fx.run("check-inputs")
        self.assertEqual(code, 1, out + err)
        self.assertRegex(err, pattern)
        self.assertIn("RESULT: FAIL (check-inputs)", out)
        self.assertEqual(os.listdir(self.fx.root / "results" / "sessionablation"), [])

    def test_check_inputs_passes(self):
        code, out, err = self.fx.run("check-inputs")
        self.assertEqual(code, 0, err)
        self.assertIn("S12 subjects / with bona fide 03 per fold: 4/1, 5/1, 5/2, 4/2; arm B 110 per cell; arm A* balanced 74,73,73,74,73,73; "
                      "dev 30 per cell", out)
        self.assertIn("RESULT: PASS (check-inputs)", out)

    def test_folds_config_sha_mismatch(self):
        self.fx.cfg["inputs"]["folds_config"]["sha256"] = "0" * 64
        self.fx.write_config()
        self.check_fails("fold schema configuration .*differs from the configuration")

    def test_m2_manifest_changed(self):
        p = self.fx.mdir / "nuaa_official_test.csv"
        p.write_bytes(p.read_bytes() + b"")
        rows = p.read_text().splitlines()
        p.write_text("\n".join(rows[:-1]) + "\n")
        self.check_fails("manifest official_test .*differs from the fold schema configuration")

    def test_fold_schema_not_pass(self):
        st = json.loads((self.fx.frun / "status.json").read_text())
        st["status"] = "FAIL"
        (self.fx.frun / "status.json").write_text(json.dumps(st))
        self.check_fails("the run is not PASS")

    def test_fold_schema_anchor_differs(self):
        self.fx.cfg["inputs"]["fold_schema"]["restricted_sha256sums"] = "1" * 64
        self.fx.write_config()
        self.check_fails("anchors differ from the configuration")

    def test_restricted_manifest_changed_without_sums(self):
        p = self.fx.frun / "restricted" / "fold2_dev.csv"
        p.write_text(p.read_text().replace(",0\n", ",1\n", 1))
        self.check_fails("fold2_dev.csv differs from its SHA256SUMS")

    def test_restricted_manifest_row_not_official(self):
        text = (self.fx.frun / "restricted" / "fold1_fitting.csv").read_text().splitlines()
        f = text[1].split(",")
        f[7] = str(int(f[7]) + 1)                       # bytes diversi dalla riga ufficiale
        text[1] = ",".join(f)
        self.fx.tamper("restricted/fold1_fitting.csv", ("\n".join(text) + "\n").encode())
        self.check_fails("not in the official lists with the same label")

    def test_restricted_manifest_wrong_role(self):
        text = (self.fx.frun / "restricted" / "fold3_dev.csv").read_text().replace(",dev,", ",fitting,", 1)
        self.fx.tamper("restricted/fold3_dev.csv", text.encode())
        self.check_fails("fold or role different from its file")

    def test_fitting_manifest_with_a_test_subject(self):
        # una riga del fitting del fold 1 sostituita con un frame di un soggetto del test del fold 1 (1101), stessa cella
        text = (self.fx.frun / "restricted" / "fold1_fitting.csv").read_text().splitlines()
        o = next(r for r in self.fx.ff.train if r["subject"] == "1101" and r["session"] == "01" and r["label"] == "1")
        f = text[1].split(",")
        text[1] = ",".join([f[0], f[1], o["key"], o["label"], o["subject"], o["session"], o["sha256"], o["bytes"], f[8]])
        self.fx.tamper("restricted/fold1_fitting.csv", ("\n".join(text) + "\n").encode())
        self.check_fails("fold1_fitting.csv: a subject outside the fitting folds")

    def test_test_manifest_incomplete(self):
        text = (self.fx.frun / "restricted" / "fold4_test.csv").read_text().splitlines()
        self.fx.tamper("restricted/fold4_test.csv", ("\n".join(text[:-1]) + "\n").encode())
        self.check_fails("every official frame exactly once")

    def test_schema_public_disagrees_with_recount(self):
        doc = json.loads((self.fx.frun / "public" / "fold_schema_public.json").read_text())
        doc["per_fold"]["2"]["roles"]["dev"]["availability"]["03/attack"] += 1
        self.fx.tamper("public/fold_schema_public.json", (json.dumps(doc, indent=1) + "\n").encode())
        self.check_fails("availability of fold 2, dev, 03/attack differs from the recount")

    def test_schema_public_manifest_sha_differs(self):
        doc = json.loads((self.fx.frun / "public" / "fold_schema_public.json").read_text())
        doc["per_fold"]["1"]["roles"]["test"]["manifest_sha256"] = "2" * 64
        self.fx.tamper("public/fold_schema_public.json", (json.dumps(doc, indent=1) + "\n").encode())
        self.check_fails("manifest SHA-256 of fold 1, test differs from the restricted manifest")

    def test_sha256sums_file_differs_from_pin(self):
        p = self.fx.frun / "restricted" / "SHA256SUMS"
        p.write_text(p.read_text() + "3" * 64 + "  extra.csv\n")      # riga valida in più: solo il pin la rifiuta
        self.check_fails("restricted/SHA256SUMS: SHA-256 differs from the configuration")

    def test_schema_public_roles_differ(self):
        doc = json.loads((self.fx.frun / "public" / "fold_schema_public.json").read_text())
        doc["per_fold"]["1"]["dev_fold"] = 3
        self.fx.tamper("public/fold_schema_public.json", (json.dumps(doc, indent=1) + "\n").encode())
        self.check_fails("roles of fold 1 differ from the rule")

    def test_expected_test_frames(self):
        self.fx.cfg["expected"]["test_frames"][0] += 1
        self.fx.write_config()
        self.check_fails("held-out fold 1: 652 test frames, expected 653")

    def test_config_values_differ_from_rules(self):
        good = json.loads(json.dumps(self.fx.cfg))
        for change, pattern in (
            (lambda c: (c["design"]["arm_b"].update(per_cell=100), c["design"]["arm_a_star"].update(total=400, allocation=SA.allocate(400, CELLS, ORDER))),
             "arm_b.per_cell = 100 differs from the rule \\(110\\)"),
            (lambda c: c["design"]["arm_a_star"]["allocation"].update({"01/bona_fide": 73, "03/attack": 74}), "arm_a_star allocation or variant differ"),
            (lambda c: (c["design"]["dev"].update(per_cell=20), c["design"]["dev_mix"].update(total=80, allocation=SA.allocate(80, CELLS, ORDER))),
             "dev.per_cell = 20 differs from the rule \\(30\\)"),
            (lambda c: c["design"]["dev_mix"]["allocation"].update({"01/bona_fide": 21, "03/attack": 19}), "dev_mix allocation or variant differ"),
            (lambda c: c["design"]["arm_b_prime"]["allocation"].update({"01/bona_fide": 73}), "arm_b_prime.allocation must equal"),
            (lambda c: c["design"].update(rounding_step=200), "smallest availability \\(117\\) is below the rounding step"),
            (lambda c: c["design"].update(rounding_step=50), "arm_b.per_cell = 110 differs from the rule \\(100\\)"),
        ):
            self.fx.cfg = json.loads(json.dumps(good))
            change(self.fx.cfg)
            self.fx.write_config()
            self.check_fails(pattern)

    def test_config_fields_and_rules(self):
        good = json.loads(json.dumps(self.fx.cfg))
        for change, pattern in (
            (lambda c: c.update(schema="other/1"), "schema must be"),
            (lambda c: c.pop("declarations"), "declarations is missing"),
            (lambda c: c["design"].update(s12="all fitting subjects"), "design.s12 must state the documented rule"),
            (lambda c: c["design"].update(sampling="first n"), "design.sampling must state the documented rule"),
            (lambda c: c["design"].update(nesting="independent samples"), "design.nesting must state the documented rule"),
            (lambda c: c["design"].pop("arm_b_prime"), "arm_b_prime needs the four session 01-02 cells"),
            (lambda c: c["design"].update(remainder_order=CELLS[:5]), "remainder_order must list the six cells"),
            (lambda c: c["design"]["arm_a_star"].update(total=441), "must equal the arm B total"),
            (lambda c: c["design"]["dev_mix"].update(variant="other"), "variant must be balanced or unbalanced"),
            (lambda c: c["inputs"]["folds_config"].update(path="/abs/x.json"), "must be a relative path"),
            (lambda c: c["inputs"]["fold_schema"].update(run_id="x"), "fold schema run id"),
            (lambda c: c.update(train_seeds=[1, 1]), "distinct non-negative integers"),
            (lambda c: c["budget"].update(reference_train_seconds=[]), "non-empty list of positive seconds"),
            (lambda c: c["expected"].update(test_frames=[1, 2]), "one count per fold"),
        ):
            self.fx.cfg = json.loads(json.dumps(good))
            change(self.fx.cfg)
            self.fx.write_config()
            self.check_fails(pattern)

    def test_budget_over_cap(self):
        self.fx.cfg["budget"]["wall_time_cap_minutes"] = 10
        self.fx.write_config()
        self.check_fails("upper estimate .* exceeds the cap of 10 min")

    def test_symlinked_manifest_refused(self):
        p = self.fx.mdir / "nuaa_official_train.csv"
        real = self.fx.mdir / "real_train.csv"
        p.rename(real)
        p.symlink_to(real)
        self.check_fails("symbolic link")


class TestRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = Fixture()
        code, out, err = cls.shared.run("run")
        assert code == 0, err
        cls.srun = cls.shared.root / "results" / "sessionablation" / RUN_ID

    @classmethod
    def tearDownClass(cls):
        cls.shared.close()

    def fresh(self) -> Fixture:
        fx = Fixture()
        self.addCleanup(fx.close)
        return fx

    def test_outputs_complete_and_anchored(self):
        run = self.srun
        self.assertEqual(json.loads((run / "status.json").read_text())["status"], "PASS")
        self.assertEqual(sorted(os.listdir(run / "public")), sorted(SA.PUBLIC_OUTPUTS))
        self.assertEqual(sorted(os.listdir(run / "restricted")), sorted(SA.restricted_outputs(4)))
        for sub in ("public", "restricted"):
            listed = {}
            for line in (run / sub / "SHA256SUMS").read_text().splitlines():
                h, name = line.split("  ")
                listed[name] = h
                self.assertEqual(sha((run / sub / name).read_bytes()), h)
            self.assertEqual(sorted(listed), sorted(set(os.listdir(run / sub)) - {"SHA256SUMS"}))
        anchor = (run / "restricted" / "public_SHA256SUMS.sha256").read_text().split()[0]
        st = json.loads((run / "status.json").read_text())
        self.assertEqual(anchor, sha((run / "public" / "SHA256SUMS").read_bytes()))
        self.assertEqual(st["public_sha256sums"], anchor)
        self.assertEqual(st["restricted_sha256sums"], sha((run / "restricted" / "SHA256SUMS").read_bytes()))

    def test_public_counts_and_manifests(self):
        doc = json.loads((self.srun / "public" / "session_ablation_public.json").read_text())
        self.assertEqual(doc["design"]["arm_b_per_cell"], N_ARM_B)
        self.assertEqual(doc["design"]["arm_a_star_allocation"], ALLOC_A)
        self.assertEqual(doc["design"]["test_frames"], TEST_FRAMES)
        f1 = doc["per_fold"]["1"]
        self.assertEqual(f1["subjects"], {"test": 4, "dev": 4, "dev_with_sessions_01_02": 2, "fitting": 8, "s12": 4,
                                          "s12_with_bona_fide_03": 1, "s12_with_attack_03": 4, "fitting_without_sessions_01_02": 4})
        self.assertEqual(f1["availability"]["s12"]["01/bona_fide"], {"subjects": 3, "status": "published", "frames": 123})
        self.assertEqual(f1["availability"]["s12"]["03/bona_fide"], {"subjects": 1, "status": SA.RESTRICTED_NOTE["one"], "frames": None,
                                                                     "at_least": 73})
        self.assertEqual(f1["feasible"], {"arm_b": True, "arm_a_star": True, "dev_common": True, "dev_mix_alternative": True})
        self.assertEqual(f1["arm_a_reference"]["fitting"]["total"], 6 * NF_FIT)
        for i in range(1, 5):
            for name, m in doc["per_fold"][str(i)]["proposed"].items():
                if name == "test_unchanged":
                    self.assertEqual(m["manifest_sha256"], sha((self.shared.frun / "restricted" / f"fold{i}_test.csv").read_bytes()))
                    continue
                text = (self.srun / "restricted" / m["manifest"]).read_text()
                self.assertEqual(sha(text.encode()), m["manifest_sha256"])
                rows = list(csv.DictReader(io.StringIO(text)))
                self.assertEqual(len(rows), m["total"])
                self.assertEqual([r["key"] for r in rows], sorted(r["key"] for r in rows))
        md = (self.srun / "public" / "session_ablation_public.md").read_text()
        self.assertIn("| S12 available | 123 | 117 | 150 | 120 | >=73 R1 | 197 | 510 | Rd |", md)
        self.assertIn("| arm B′ (optional) (subj.) | 74 (3) | 73 (4) | 73 (4) | 74 (4) | - | - | 294 | 294 |", md)
        self.assertIn("Optional arm B′ (not in the base budget): 12 trainings of 294 fitting + 120 dev frames", md)
        self.assertTrue(all(len(line) <= 120 for line in md.splitlines()), "a Markdown line longer than 120 characters")

    def test_public_has_no_identifier_or_restricted_value(self):
        restricted = json.loads((self.srun / "restricted" / "counts_restricted.json").read_text())
        doc = json.loads((self.srun / "public" / "session_ablation_public.json").read_text())
        held = {q: v for q, v in restricted["quantities"].items() if v["status"] != "published"}
        self.assertTrue(held)
        for q, v in held.items():
            fold, row, cell = q.split(".", 2)
            entry = doc["per_fold"][fold[4:]]["availability"][row][cell]
            self.assertIsNone(entry["frames"], q)
            self.assertEqual(entry["status"], v["status"])
        for name in ("session_ablation_public.json", "session_ablation_public.md"):
            text = (self.srun / "public" / name).read_text()
            for s in SUBJECTS:
                self.assertNotIn(f"{s}_{s}_", text)
                self.assertNotIn(f'"{s}"', text)
                self.assertNotIn(f"subject {s}", text)
            self.assertNotIn(".jpg", text)

    def test_restricted_subject_lists(self):
        rows = list(csv.DictReader(io.StringIO((self.srun / "restricted" / "subjects.csv").read_text())))
        s12 = {(int(r["held_out_fold"]), r["subject"]) for r in rows if r["s12"] == "1"}
        self.assertEqual(s12, {(i, s) for i, v in S12.items() for s in v})
        r1103 = next(r for r in rows if r["held_out_fold"] == "1" and r["subject"] == "1103")
        self.assertEqual([int(r1103[c.replace("/", "_")]) for c in CELLS], [44, 30, 40, 30, 85, 52])

    def test_run_without_authorization_refused(self):
        fx = self.fresh()
        code, out, err = fx.run("run", authorized=False)
        self.assertEqual(code, 1)
        self.assertIn("--i-am-authorized", err)
        self.assertEqual(os.listdir(fx.root / "results" / "sessionablation"), [])

    def test_no_old_pass_after_fail(self):
        fx = self.fresh()
        self.assertEqual(fx.run("run")[0], 0)
        run = fx.root / "results" / "sessionablation" / RUN_ID
        fx.cfg["budget"]["wall_time_cap_minutes"] = 10
        fx.write_config()
        code, out, err = fx.run("run")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads((run / "status.json").read_text())["status"], "FAIL")
        self.assertEqual(os.listdir(run / "public"), [])
        self.assertEqual(os.listdir(run / "restricted"), [])

    def test_no_old_pass_after_input_failure(self):
        fx = self.fresh()
        self.assertEqual(fx.run("run")[0], 0)
        run = fx.root / "results" / "sessionablation" / RUN_ID
        st = json.loads((fx.frun / "status.json").read_text())
        st["status"] = "FAIL"
        (fx.frun / "status.json").write_text(json.dumps(st))
        code, out, err = fx.run("run")
        self.assertEqual(code, 1)
        self.assertIn("not PASS", err)
        self.assertEqual(json.loads((run / "status.json").read_text())["status"], "FAIL")
        self.assertEqual(os.listdir(run / "public"), [])

    def test_public_identifier_refused(self):
        fx = self.fresh()
        fx.cfg["declarations"] = {"note": "subject 1103"}
        fx.write_config()
        code, out, err = fx.run("run")
        self.assertEqual(code, 1)
        self.assertIn("public outputs refused", err)
        run = fx.root / "results" / "sessionablation" / RUN_ID
        self.assertEqual(json.loads((run / "status.json").read_text())["status"], "FAIL")
        self.assertFalse((run / "public").exists() and os.listdir(run / "public"))

    def test_disclosure_failure_refuses_outputs(self):
        fx = self.fresh()
        with mock.patch.object(SA, "disclosure_problems", lambda *a: ["forced"]):
            code, out, err = fx.run("run")
        self.assertEqual(code, 1)
        self.assertIn("public counts refused: forced", err)
        run = fx.root / "results" / "sessionablation" / RUN_ID
        self.assertEqual(json.loads((run / "status.json").read_text())["status"], "FAIL")

    def test_foreign_file_refused(self):
        fx = self.fresh()
        run = fx.root / "results" / "sessionablation" / RUN_ID
        (run / "restricted").mkdir(parents=True)
        (run / "restricted" / "other.csv").write_text("x")
        code, out, err = fx.run("run")
        self.assertEqual(code, 1)
        self.assertIn("not an output of this command", err)
        self.assertEqual((run / "restricted" / "other.csv").read_text(), "x")
        self.assertFalse((run / "status.json").exists())

    def test_protected_out_roots_refused(self):
        fx = self.fresh()
        for rel in ("results/folds", "results/m2"):
            (fx.root / rel).mkdir(parents=True, exist_ok=True)
            code, out, err = fx.run("run", out_root=fx.root / rel)
            self.assertEqual(code, 1, rel)
            self.assertFalse((fx.root / rel / RUN_ID).exists())

    def test_symlinked_out_root_refused(self):
        fx = self.fresh()
        target = fx.tmp / "elsewhere"
        target.mkdir()
        (fx.root / "results" / "link").symlink_to(target)
        code, out, err = fx.run("run", out_root="results/link")
        self.assertEqual(code, 1)
        self.assertIn("symbolic link", err)
        self.assertEqual(os.listdir(target), [])

    def test_run_inside_inputs_refused(self):
        fx = self.fresh()
        code, out, err = fx.run("run", out_root=fx.mdir)
        self.assertEqual(code, 1)
        self.assertIn("inside the folder of manifests", err)
        self.assertFalse((fx.mdir / RUN_ID).exists())

    def test_failure_after_public_written_leaves_nothing(self):
        fx = self.fresh()
        real_write = SA.mw.write_all

        def failing_write(out, files, newline=None):
            if Path(out).name == SA.RESTRICTED:
                raise OSError("disk full (simulated)")
            return real_write(out, files, newline)

        with mock.patch.object(SA.mw, "write_all", failing_write):
            code, out, err = fx.run("run")
        self.assertEqual(code, 1)
        run = fx.root / "results" / "sessionablation" / RUN_ID
        self.assertEqual(json.loads((run / "status.json").read_text())["status"], "FAIL")
        self.assertEqual(os.listdir(run / "public"), [])
        self.assertEqual(os.listdir(run / "restricted"), [])

    def test_dirty_tree_gets_dev_suffix(self):
        self.assertEqual(SA.run_id_for("20261104", {"commit": "abcdef0", "dirty": True}), RUN_ID + "-dev")
        self.assertEqual(SA.run_id_for("20261104", {"commit": "abcdef0", "dirty": False}), RUN_ID)
        with self.assertRaisesRegex(SA.Failure, "YYYYMMDD"):
            SA.run_id_for("2026-11-04", STATE)


class TestRealInputs(unittest.TestCase):
    def test_real_inputs_check(self):
        cfg_p = REAL_ROOT / SA.DEFAULT_CONFIG
        if not cfg_p.is_file() or not (REAL_ROOT / "results" / "m2" / "manifests").is_dir() \
                or not (REAL_ROOT / "results" / "folds" / "20261103-FOLDS-67b7f3d").is_dir():
            skip_missing(self, "real inputs")
        cfg, _ = SA.load_config(cfg_p)
        a = argparse.Namespace(manifests="results/m2/manifests", folds_root="results/folds")
        inp, des, base, dec, leaks, bud = SA.compute(cfg, REAL_ROOT, a)
        self.assertEqual([len(f["s12"]) for f in des["folds"].values()], [4, 5, 5, 4])
        self.assertEqual([sum(1 for s in f["s12"] if des["count"][(s, "03/bona_fide")]) for f in des["folds"].values()], [1, 1, 2, 2])
        self.assertEqual((des["min_b"], des["n"], des["min_dev"], des["dev_n"]), (361, 350, 117, 100))
        self.assertEqual([des["alloc_a"][c] for c in CELLS], [234, 233, 233, 234, 233, 233])
        self.assertEqual(des["variant_a"], "balanced")
        self.assertEqual([len(f["test_rows"]) for f in des["folds"].values()], [3242, 3114, 3439, 2819])
        self.assertEqual(bud["trainings"], 24)
        self.assertEqual(SA.disclosure_problems(des, base, dec), [])


def main() -> int:
    before = snapshot(ROOT / "results", ROOT / "export", ROOT / "delivery")
    result = unittest.main(argv=[sys.argv[0]] + sys.argv[1:], exit=False, verbosity=1).result
    after = snapshot(ROOT / "results", ROOT / "export", ROOT / "delivery")
    changed = before != after
    skipped = [f"{t.id()}: {r}" for t, r in result.skipped]
    print(f"tests run: {result.testsRun}, failures: {len(result.failures)}, errors: {len(result.errors)}, skipped: {len(skipped)}")
    for s in skipped:
        print(f"SKIPPED {s}")
    if changed:
        print("ERROR: results/, export/ or delivery/ changed during the tests")
    ok = result.wasSuccessful() and not changed
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
