"""
Test di scripts/nuaa_subject_folds.py (schema dei fold per soggetto su NUAA), offline e deterministici.

Fixture sintetiche con identificativi fuori da quelli NUAA (1101–1116) e struttura diversa da quella reale (strati
2/6/6/2): 8 soggetti con frame nelle sessioni 01–02, bona fide nella sessione 03 per 8 soggetti, un soggetto con solo
bona fide nella 03, due con solo attacchi nella 03, uno senza la sessione 02, uno senza bona fide nella 01; manifest
della cache dentro le liste. Risposte calcolate a mano (commenti accanto ai valori attesi): strati, assegnazione
bilanciata, ruoli (dev = fold successivo, fitting = gli altri due), campionamento `spread` per cella, disgiunzione
dei ruoli, disponibilità, conteggi pubblici, frame «in cache» e «da estrarre», completezza dei SHA256SUMS. Percorsi
d'errore: fold sotto il minimo di soggetti per proprietà, cella con meno frame del conteggio fissato, SHA-256,
conteggi e attesi diversi dalla configurazione, colonne mancanti, nome del file in disaccordo con soggetto o
sessione, etichetta o sessione fuori dominio, sha256 o bytes malformati, chiave duplicata, contenuto duplicato
dentro una lista, liste ufficiali che condividono un contenuto, lista di test con la sessione 01, manifest della
cache con chiavi condivise o con un campo diverso dalla riga ufficiale, collegamento simbolico (manifest, radice
delle uscite), configurazione senza i campi obbligatori o con schema, regola, file, seed o conteggi non ammessi,
ingresso dentro la cartella del run o run dentro la cartella dei manifest → FAIL senza uscite; nessun identificativo
nel pubblico (chiave o soggetto, anche in un campo solo JSON); nessun PASS vecchio dopo un FAIL (anche se il manifest
è guasto); cartella di uscita estranea o protetta → rifiuto prima di qualunque modifica. Un test usa i manifest veri
(sola lettura, uscite in una cartella temporanea) e si salta solo se mancano (ingresso mancante dichiarato in
MISSING_INPUT). Alla fine si controlla che results/, export/ e delivery/ del repository non siano cambiati (i log dei
test vanno scritti fuori da quelle cartelle e copiati dopo).

  .venv/bin/python scripts/check_nuaa_subject_folds.py        # uscita 1 se un test fallisce

Variabile facoltativa FOLDS_INPUT_ROOT: radice degli ingressi veri (predefinita: la radice del repository; la usa il
controllo delle mutazioni, che esegue questa suite su una copia del codice).
"""
import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nuaa_subject_folds as NF  # noqa: E402

REAL_ROOT = Path(os.environ.get("FOLDS_INPUT_ROOT") or ROOT)
STATE = {"commit": "abcdef0", "dirty": False}
RUN_ID = "20261103-FOLDS-abcdef0"
MISSING_INPUT = {"real manifests": "the restricted M2 manifests are not present (results/m2/manifests)"}


def skip_missing(test: unittest.TestCase, what: str):
    """Unico modo di saltare un test di questa suite: un ingresso mancante dichiarato in MISSING_INPUT."""
    test.skipTest(f"missing input: {MISSING_INPUT[what]}")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ============================================================================ fixture

# soggetto -> (frame per sessione 01 (bf, att), 02 (bf, att), 03 (bf, att)); identificativi finti, struttura finta
SUBJECTS = {
    "1101": ((5, 4), (5, 4), (0, 6)), "1102": ((3, 4), (0, 0), (0, 6)), "1103": ((5, 4), (5, 4), (6, 6)),
    "1104": ((0, 4), (5, 4), (0, 6)), "1105": ((5, 4), (5, 4), (0, 6)), "1106": ((4, 4), (5, 4), (0, 6)),
    "1107": ((5, 4), (5, 4), (6, 5)), "1108": ((4, 4), (4, 4), (0, 6)), "1109": ((0, 0), (0, 0), (6, 5)),
    "1110": ((0, 0), (0, 0), (6, 6)), "1111": ((0, 0), (0, 0), (6, 0)), "1112": ((0, 0), (0, 0), (6, 6)),
    "1113": ((0, 0), (0, 0), (3, 6)), "1114": ((0, 0), (0, 0), (6, 6)), "1115": ((0, 0), (0, 0), (0, 6)),
    "1116": ((0, 0), (0, 0), (0, 5)),
}
STRATA = {"1103": "A", "1107": "A", "1101": "B", "1102": "B", "1104": "B", "1105": "B", "1106": "B", "1108": "B",
          "1109": "C", "1110": "C", "1111": "C", "1112": "C", "1113": "C", "1114": "C", "1115": "D", "1116": "D"}
# a mano, regola bilanciata (fold meno pieno; a parità meno soggetti con la stessa proprietà; poi indice):
# A: 1103→1, 1107→2 (fold vuoti); B: 1101→3, 1102→4, poi tutti a 1: 1104→1 (ogni fold ha 1 soggetto 01-02, parità → 1),
# 1105→2, 1106→3, 1108→4; C: 1109→3 (fold 3 e 4 senza bona fide 03), 1110→4, 1111→1, 1112→2, 1113→3 (fold 3 e 4 con 1
# bona fide 03 contro 2), 1114→4; D: 1115→1, 1116→2 (fold meno pieni)
FOLDS = {1: ["1103", "1104", "1111", "1115"], 2: ["1107", "1105", "1112", "1116"], 3: ["1101", "1106", "1109", "1113"],
         4: ["1102", "1108", "1110", "1114"]}
FIT, DEV = 3, 2         # conteggi per cella della fixture (disponibilità minima: fitting 12, dev 4)


def make_rows(subjects=SUBJECTS) -> tuple:
    """(righe ufficiali train, righe ufficiali test) con nomi NUAA e SHA-256 finti distinti."""
    train, test = [], []
    for s, sessions in subjects.items():
        for ses, (nb, na) in zip(("01", "02", "03"), sessions):
            for lab, n in (("0", nb), ("1", na)):
                for i in range(n):
                    name = f"{s}_{s}_00_0{lab}_{ses}_{i}.jpg"
                    key = f"{'real' if lab == '0' else 'attack'}/{name}"
                    row = {"key": key, "label": lab, "subject": s, "glasses": "00", "pos": f"0{lab}", "session": ses, "pic": str(i),
                           "official_split": "train" if ses != "03" else "test", "sha256": sha(key.encode()), "bytes": str(1000 + i)}
                    (test if ses == "03" else train).append(row)
    return train, test


def csv_bytes(rows, columns=NF.COLUMNS) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(columns), lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def counts(rows) -> dict:
    return {"rows": len(rows), "n_bona_fide": sum(r["label"] == "0" for r in rows), "n_attack": sum(r["label"] == "1" for r in rows),
            "n_subjects": len({r["subject"] for r in rows})}


class Fixture:
    """Cartella temporanea con repository finto (git sostituito), manifest sintetici e configurazione coerente."""

    def __init__(self, subjects=SUBJECTS, fit=FIT, dev=DEV):
        self.tmp = Path(tempfile.mkdtemp(prefix="folds_"))
        self.root = self.tmp / "repo"
        (self.root / "results" / "folds").mkdir(parents=True)
        self.mdir = self.root / "manifests"
        self.mdir.mkdir()
        train, test = make_rows(subjects)
        self.train, self.test = train, test
        # cache: i frame con pic 0 e 1 delle liste ufficiali in tre manifest disgiunti (es = due soggetti di training)
        cache_train = [r for r in train if r["pic"] in ("0", "1")]
        es = [r for r in cache_train if r["subject"] in ("1104", "1108")]
        fitting = [r for r in cache_train if r["subject"] not in ("1104", "1108")]
        cache_test = [r for r in test if r["pic"] in ("0", "1")]
        self.manifests = {"official_train": train, "official_test": test, "cnn_train": fitting, "cnn_es": es, "cnn_test": cache_test}
        self.files = {"official_train": "nuaa_official_train.csv", "official_test": "nuaa_official_test.csv", "cnn_train": "nuaa_cnn_train.csv",
                      "cnn_es": "nuaa_cnn_early_stopping.csv", "cnn_test": "nuaa_cnn_test.csv"}
        self.cache_keys = {r["key"] for m in ("cnn_train", "cnn_es", "cnn_test") for r in self.manifests[m]}
        for k, rows in self.manifests.items():
            (self.mdir / self.files[k]).write_bytes(csv_bytes(rows))
        self.cfg = self.make_config(fit, dev)
        self.cfg_path = self.root / "delivery" / "folds" / "nuaa_subject_folds.json"
        self.cfg_path.parent.mkdir(parents=True)
        self.write_config()
        for f in NF.CODE_FILES:
            (self.root / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / f, self.root / f)

    def make_config(self, fit, dev) -> dict:
        return {
            "schema": NF.CONFIG_SCHEMA, "status": "test", "task": "test",
            "manifests": {k: {"file": self.files[k], "sha256": sha((self.mdir / self.files[k]).read_bytes()), **counts(rows)}
                          for k, rows in self.manifests.items()},
            "expected": {"n_subjects": len({r["subject"] for r in self.train + self.test}), "n_images": len(self.train) + len(self.test),
                         "n_cache_frames": len(self.cache_keys)},
            "folds": {"k": 4, "strata": list(NF.STRATA), "dealing": NF.DEALING,
                      "dev_fold": "fold (k mod K) + 1 for the held-out fold k", "fitting_folds": "the other K - 2 folds"},
            "sampling": {"fitting_per_cell": fit, "dev_per_cell": dev,
                         "rule": "spread(sorted file names of the cell, n): one every len/n, from the first", "test": "all frames of the held-out subjects"},
            "train_seeds": [42, 43, 44], "architecture": "LivenessNet", "training": {"max_epochs": 12}, "threshold": {"set": "dev"},
            "metrics": {"x": 1}, "frames": {"cache": "test"}, "budget": {"estimate": "test", "wall_time_cap_minutes": 1}, "declarations": {"d": 1},
        }

    def write_config(self):
        self.cfg_path.write_text(json.dumps(self.cfg, indent=1) + "\n", encoding="utf-8")

    def rewrite_manifest(self, name: str, rows, columns=NF.COLUMNS, fix_sha=True):
        self.manifests[name] = rows
        b = csv_bytes(rows, columns)
        (self.mdir / self.files[name]).write_bytes(b)
        if fix_sha:
            self.cfg["manifests"][name] = {"file": self.files[name], "sha256": sha(b), **counts(rows)}
            self.write_config()

    def args(self, cmd, out_root=None, date="20261103", manifests="manifests"):
        return [cmd, "--config", str(self.cfg_path.relative_to(self.root)), "--manifests", str(manifests),
                "--out-root", str(out_root or (self.root / "results" / "folds")), "--date", date]

    def run(self, cmd, **kw) -> tuple:
        """(codice di uscita, stdout, stderr) con ROOT e lo stato git sostituiti."""
        out, err = io.StringIO(), io.StringIO()
        with mock_root(self.root), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = NF.main(self.args(cmd, **kw))
        return code, out.getvalue(), err.getvalue()

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


@contextlib.contextmanager
def mock_root(root: Path):
    old_root, old_state = NF.ROOT, NF.REPO_STATE
    NF.ROOT, NF.REPO_STATE = root, lambda r: dict(STATE)
    try:
        yield
    finally:
        NF.ROOT, NF.REPO_STATE = old_root, old_state


def snapshot(*dirs) -> dict:
    out = {}
    for d in dirs:
        for p in sorted(Path(d).rglob("*")):
            if p.is_file():
                out[str(p)] = (p.stat().st_size, p.stat().st_mtime_ns)
    return out


def cell_count(subjects, ses, lab) -> int:
    i = ("01", "02", "03").index(ses)
    return sum(SUBJECTS[s][i][0 if lab == "0" else 1] for s in subjects)


# ============================================================================ test

class TestSchema(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture()
        self.addCleanup(self.fx.close)

    def test_strata_and_properties_by_hand(self):
        props = NF.subject_properties(self.fx.train + self.fx.test)
        self.assertEqual({s: v["stratum"] for s, v in props.items()}, STRATA)
        self.assertEqual(props["1111"], {"stratum": "C", "sessions_01_02": False, "bona_fide_03": True, "attack_03": False})
        self.assertEqual(props["1104"], {"stratum": "B", "sessions_01_02": True, "bona_fide_03": False, "attack_03": True})
        self.assertEqual(NF.strata_of(self.fx.train + self.fx.test), STRATA)

    def test_balanced_dealing_by_hand(self):
        props = NF.subject_properties(self.fx.train + self.fx.test)
        self.assertEqual(NF.deal_folds(props, 4), FOLDS)

    def test_dealing_tie_breaks(self):
        # due A, un B e un C su 2 fold: il secondo A va al fold 2 (meno pieno); il B al fold con meno soggetti 01-02 a
        # parità di dimensione (entrambi 1 → indice 1); il C al fold 2 (meno pieno)
        props = {"a1": {"stratum": "A", "sessions_01_02": True, "bona_fide_03": True, "attack_03": True},
                 "a2": {"stratum": "A", "sessions_01_02": True, "bona_fide_03": True, "attack_03": True},
                 "b1": {"stratum": "B", "sessions_01_02": True, "bona_fide_03": False, "attack_03": True},
                 "c1": {"stratum": "C", "sessions_01_02": False, "bona_fide_03": True, "attack_03": True}}
        self.assertEqual(NF.deal_folds(props, 2), {1: ["a1", "b1"], 2: ["a2", "c1"]})
        # la proprietà condivisa decide a parità di dimensione: fold 1 {A}, fold 2 {B}; il secondo B condivide 01-02 con
        # entrambi (1 e 1) → indice 1; il C condivide bona fide 03 solo con A (fold 1: 1, fold 2: 0) → fold 2
        props2 = {"a1": props["a1"], "b1": props["b1"], "b2": dict(props["b1"]), "c1": props["c1"]}
        self.assertEqual(NF.deal_folds(props2, 2), {1: ["a1", "b2"], 2: ["b1", "c1"]})

    def test_spread_by_hand(self):
        self.assertEqual(NF.spread(["c", "a", "b", "d", "e"], 2), ["a", "c"])          # passo 2.5: indici 0, 2
        self.assertEqual(NF.spread(["b", "a"], 5), ["a", "b"])                         # k >= len: tutto, ordinato
        self.assertEqual(NF.spread(["d", "c", "b", "a"], 3), ["a", "b", "c"])          # passo 1.33: indici 0, 1, 2

    def test_roles_and_sampling_by_hand(self):
        mans = NF.load_manifests(self.fx.cfg, self.fx.mdir)
        schema = NF.build_schema(self.fx.cfg, mans)
        self.assertEqual(schema["folds"], FOLDS)
        pf = schema["per_fold"][1]
        self.assertEqual((pf["dev_fold"], pf["fitting_folds"]), (2, [3, 4]))
        self.assertEqual(pf["subjects"]["fitting"], set(FOLDS[3] + FOLDS[4]))
        self.assertEqual(pf["subjects"]["dev"], set(FOLDS[2]))
        self.assertEqual(pf["subjects"]["test"], set(FOLDS[1]))
        fit = pf["manifests"]["fitting"]
        self.assertEqual(len(fit), 6 * FIT)
        for ses in NF.SESSIONS:
            for lab in NF.LABELS:
                cell = [r for r in fit if r["session"] == ses and r["label"] == lab]
                self.assertEqual(len(cell), FIT, (ses, lab))
                names = sorted(r["key"].split("/", 1)[1] for r in self.fx.train + self.fx.test
                               if r["subject"] in pf["subjects"]["fitting"] and r["session"] == ses and r["label"] == lab)
                self.assertEqual(sorted(r["key"].split("/", 1)[1] for r in cell), sorted(NF.spread(names, FIT)))
        self.assertEqual(len(pf["manifests"]["dev"]), 6 * DEV)
        test = pf["manifests"]["test"]
        self.assertEqual(len(test), sum(1 for r in self.fx.train + self.fx.test if r["subject"] in FOLDS[1]))
        self.assertEqual([r["key"] for r in test], sorted(r["key"] for r in test))                 # ordinati per chiave
        keys = [r["key"] for role in NF.ROLES for r in pf["manifests"][role]]
        self.assertEqual(len(keys), len(set(keys)))
        # disponibilità a mano: fold 1 = 1103 (5,4)(5,4)(6,6), 1104 (0,4)(5,4)(0,6), 1111 (6,0 nella 03), 1115 (0,6 nella 03)
        self.assertEqual(pf["availability"]["test"], {"01/bona_fide": 5, "01/attack": 8, "02/bona_fide": 10, "02/attack": 8,
                                                      "03/bona_fide": 12, "03/attack": 18})
        self.assertEqual(pf["availability"]["dev"]["03/attack"], 22)                               # fold 2: 5 + 6 + 6 + 5
        self.assertEqual(pf["availability"]["fitting"]["01/bona_fide"], 16)                        # fold 3 (5 + 4) + fold 4 (3 + 4)
        for role in NF.ROLES:
            for ses in NF.SESSIONS:
                for lab in NF.LABELS:
                    self.assertEqual(pf["availability"][role][f"{ses}/{NF.CLASS_NAME[lab]}"], cell_count(pf["subjects"][role], ses, lab))

    def test_every_subject_held_out_once(self):
        mans = NF.load_manifests(self.fx.cfg, self.fx.mdir)
        schema = NF.build_schema(self.fx.cfg, mans)
        tested = [r["key"] for i in range(1, 5) for r in schema["per_fold"][i]["manifests"]["test"]]
        self.assertEqual(sorted(tested), sorted(r["key"] for r in self.fx.train + self.fx.test))
        for i in range(1, 5):
            pf = schema["per_fold"][i]
            self.assertFalse(pf["subjects"]["test"] & (pf["subjects"]["fitting"] | pf["subjects"]["dev"]))
            self.assertFalse(pf["subjects"]["fitting"] & pf["subjects"]["dev"])

    def test_fold_below_minimum_subjects_fails(self):
        subjects = dict(SUBJECTS)
        for s in ("1109", "1110", "1111", "1112", "1113"):      # restano 3 soggetti con bona fide nella 03: un fold ne avrà < 2
            subjects[s] = ((0, 0), (0, 0), (0, 6))
        fx = Fixture(subjects)
        self.addCleanup(fx.close)
        mans = NF.load_manifests(fx.cfg, fx.mdir)
        with self.assertRaisesRegex(NF.Failure, "subject\\(s\\) with bona fide frames in session 03, at least 2 required"):
            NF.build_schema(fx.cfg, mans)

    def test_fold_without_sessions_01_02_fails(self):
        subjects = dict(SUBJECTS)
        for s in ("1101", "1102", "1104", "1105", "1106"):     # restano 3 soggetti nelle sessioni 01-02
            subjects[s] = ((0, 0), (0, 0), (3, 6))
        fx = Fixture(subjects)
        self.addCleanup(fx.close)
        mans = NF.load_manifests(fx.cfg, fx.mdir)
        with self.assertRaisesRegex(NF.Failure, "with frames in sessions 01-02, at least 2 required"):
            NF.build_schema(fx.cfg, mans)

    def test_cell_shortfall_fails(self):
        fx = Fixture(fit=30)     # più dei frame disponibili in ogni cella delle sessioni 01-02
        self.addCleanup(fx.close)
        mans = NF.load_manifests(fx.cfg, fx.mdir)
        with self.assertRaisesRegex(NF.Failure, "frames available, 30 required"):
            NF.build_schema(fx.cfg, mans)


class TestInputs(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture()
        self.addCleanup(self.fx.close)

    def check_fails(self, pattern: str):
        code, out, err = self.fx.run("check-inputs")
        self.assertEqual(code, 1, err)
        self.assertRegex(err, pattern)
        self.assertIn("RESULT: FAIL (check-inputs)", out)
        self.assertEqual(sorted(os.listdir(self.fx.root / "results" / "folds")), [])

    def test_check_inputs_passes(self):
        code, out, err = self.fx.run("check-inputs")
        self.assertEqual(code, 0, err)
        self.assertIn("RESULT: PASS (check-inputs)", out)
        self.assertIn("5 manifests verified, 4 folds, 16 subjects, nothing written", out)

    def test_sha_mismatch(self):
        self.fx.cfg["manifests"]["cnn_es"]["sha256"] = "0" * 64
        self.fx.write_config()
        self.check_fails("SHA-256 .* differs from the configuration")

    def test_count_mismatch(self):
        self.fx.cfg["manifests"]["official_test"]["n_attack"] += 1
        self.fx.write_config()
        self.check_fails("n_attack = .* differs from the configuration")

    def test_expected_mismatch(self):
        good = json.loads(json.dumps(self.fx.cfg))
        for key, pattern in (("n_images", "images, expected"), ("n_subjects", "number of subjects differs"), ("n_cache_frames", "frames, expected")):
            self.fx.cfg = json.loads(json.dumps(good))
            self.fx.cfg["expected"][key] += 1
            self.fx.write_config()
            self.check_fails(pattern)
        self.fx.cfg = good
        self.fx.write_config()

    def test_missing_column(self):
        self.fx.rewrite_manifest("cnn_test", self.fx.manifests["cnn_test"], columns=[c for c in NF.COLUMNS if c != "session"])
        self.check_fails("columns must be exactly")

    def test_name_disagrees_with_session(self):
        rows = [dict(r) for r in self.fx.manifests["official_train"]]
        rows[0]["session"] = "02" if rows[0]["session"] == "01" else "01"
        self.fx.rewrite_manifest("official_train", rows)
        self.check_fails("does not carry the subject and session of its row")

    def test_name_disagrees_with_subject(self):
        rows = [dict(r) for r in self.fx.manifests["official_train"]]
        rows[0]["subject"] = "1199"
        self.fx.rewrite_manifest("official_train", rows)
        self.check_fails("does not carry the subject and session of its row")

    def test_label_or_session_outside_domain(self):
        rows = [dict(r) for r in self.fx.manifests["official_train"]]
        rows[0]["label"] = "2"
        self.fx.rewrite_manifest("official_train", rows)
        self.check_fails("label or session outside the admitted values")
        rows = [dict(r) for r in self.fx.manifests["official_test"]]
        rows[0]["session"] = "04"
        self.fx.rewrite_manifest("official_test", rows)
        self.check_fails("label or session outside the admitted values")

    def test_malformed_sha_or_bytes(self):
        rows = [dict(r) for r in self.fx.manifests["cnn_es"]]
        rows[0]["sha256"] = "xyz"
        self.fx.rewrite_manifest("cnn_es", rows)
        self.check_fails("sha256 or bytes malformed")
        rows[0]["sha256"] = "0" * 64
        rows[0]["bytes"] = "12a"
        self.fx.rewrite_manifest("cnn_es", rows)
        self.check_fails("sha256 or bytes malformed")

    def test_duplicated_key(self):
        rows = self.fx.manifests["cnn_train"]
        self.fx.rewrite_manifest("cnn_train", rows + [rows[0]])
        self.check_fails("duplicated keys")

    def test_duplicated_content_inside_a_list(self):
        rows = [dict(r) for r in self.fx.manifests["official_test"]]
        rows[1] = dict(rows[1], sha256=rows[0]["sha256"])
        self.fx.rewrite_manifest("official_test", rows)
        self.check_fails("two rows share a file SHA-256")

    def test_official_lists_share_a_file(self):
        rows = [dict(r) for r in self.fx.manifests["official_test"]]
        rows[-1] = dict(rows[-1], sha256=self.fx.manifests["official_train"][0]["sha256"])     # stesso contenuto, chiave diversa
        self.fx.rewrite_manifest("official_test", rows)
        self.check_fails("share a key or a file SHA-256")

    def test_official_train_with_session_03_refused(self):
        rows = [dict(r) for r in self.fx.manifests["official_train"]]
        rows.append(dict(self.fx.manifests["official_test"][0], key="real/1103_1103_00_00_03_99.jpg", label="0", subject="1103", sha256="2" * 64, pic="99"))
        self.fx.rewrite_manifest("official_train", rows)
        self.check_fails("official train list must hold sessions 01-02 only")

    def test_official_test_with_session_01_refused(self):
        rows = [dict(r) for r in self.fx.manifests["official_test"]]
        rows.append(dict(self.fx.manifests["official_train"][0], key="attack/1103_1103_00_01_01_99.jpg", label="1", subject="1103", session="01",
                         sha256="3" * 64, pic="99"))
        self.fx.rewrite_manifest("official_test", rows)
        self.check_fails("official test list session 03 only")

    def test_cache_key_outside_official_lists(self):
        rows = [dict(r) for r in self.fx.manifests["cnn_test"]]
        rows[0] = dict(rows[0], sha256="1" * 64)
        self.fx.rewrite_manifest("cnn_test", rows)
        self.check_fails("not in the official lists with the same values in every column")

    def test_cache_row_with_a_different_field(self):
        rows = [dict(r) for r in self.fx.manifests["cnn_train"]]
        rows[0] = dict(rows[0], bytes=str(int(rows[0]["bytes"]) + 1))
        self.fx.rewrite_manifest("cnn_train", rows)
        self.check_fails("same values in every column")

    def test_cache_manifests_share_a_key(self):
        rows = self.fx.manifests["cnn_es"] + [self.fx.manifests["cnn_train"][0]]
        self.fx.rewrite_manifest("cnn_es", rows)
        self.check_fails("cache manifests share keys")

    def test_symlink_refused(self):
        p = self.fx.mdir / self.fx.files["cnn_es"]
        real = self.fx.mdir / "real_es.csv"
        p.rename(real)
        p.symlink_to(real)
        self.check_fails("symbolic link")

    def test_config_missing_field(self):
        del self.fx.cfg["budget"]
        self.fx.write_config()
        self.check_fails("budget is missing")

    def test_config_wrong_rule_text(self):
        self.fx.cfg["folds"]["dev_fold"] = "the same fold"
        self.fx.write_config()
        self.check_fails("must state the documented rule")
        self.fx.cfg["folds"]["dev_fold"] = "fold (k mod K) + 1 for the held-out fold k"
        self.fx.cfg["folds"]["dealing"] = "round-robin"
        self.fx.write_config()
        self.check_fails("must state the documented rule")

    def test_config_schema_file_seeds_counts(self):
        good = json.loads(json.dumps(self.fx.cfg))
        for change, pattern in (
            (lambda c: c.update(schema="other/1"), "schema must be"),
            (lambda c: c["manifests"]["cnn_es"].update(file="sub/x.csv"), "file must be a bare file name"),
            (lambda c: c["manifests"]["cnn_es"].pop("sha256"), "needs file and sha256"),
            (lambda c: c["manifests"]["cnn_es"].update(rows=True), "needs the counts"),
            (lambda c: c.update(train_seeds=[42, 42]), "distinct non-negative integers"),
            (lambda c: c["sampling"].update(dev_per_cell=0), "positive integers"),
            (lambda c: c["folds"].update(k=1), "integer >= 2"),
        ):
            self.fx.cfg = json.loads(json.dumps(good))
            change(self.fx.cfg)
            self.fx.write_config()
            self.check_fails(pattern)
        self.fx.cfg = good
        self.fx.write_config()


class TestRun(unittest.TestCase):
    def setUp(self):
        self.fx = Fixture()
        self.addCleanup(self.fx.close)

    def run_ok(self) -> Path:
        code, out, err = self.fx.run("run")
        self.assertEqual(code, 0, err)
        self.assertIn("RESULT: PASS (run)", out)
        run = self.fx.root / "results" / "folds" / RUN_ID
        self.assertEqual(json.loads((run / "status.json").read_text())["status"], "PASS")
        return run

    def test_outputs_public_and_restricted(self):
        run = self.run_ok()
        self.assertEqual(sorted(os.listdir(run / "public")), sorted(NF.PUBLIC_OUTPUTS))
        self.assertEqual(sorted(os.listdir(run / "restricted")), sorted(NF.restricted_outputs(4)))
        doc = json.loads((run / "public" / "fold_schema_public.json").read_text())
        self.assertEqual(doc["folds"]["subjects_per_fold"], {"1": 4, "2": 4, "3": 4, "4": 4})
        self.assertEqual(doc["folds"]["strata_sizes"], {"A": 2, "B": 6, "C": 6, "D": 2})
        self.assertEqual(doc["folds"]["min_subjects_per_property"], 2)
        self.assertEqual(doc["budget"]["trainings"], 12)
        self.assertEqual(doc["dirty"], False)
        f1 = doc["per_fold"]["1"]
        self.assertEqual(f1["subjects_with"], {"bona_fide_03": 2, "attack_03": 3, "sessions_01_02": 2})     # 1103, 1111 / 1103, 1104, 1115 / 1103, 1104
        r1 = f1["roles"]
        self.assertEqual(r1["fitting"]["counts"]["total"]["frames"], 6 * FIT)
        self.assertEqual(r1["dev"]["counts"]["total"]["frames"], 6 * DEV)
        self.assertEqual(r1["test"]["counts"]["03/bona_fide"], {"frames": 12, "subjects": 2, "in_cache": 4, "to_extract": 8})
        self.assertEqual(r1["test"]["availability"]["01/bona_fide"], 5)
        self.assertEqual(r1["fitting"]["availability"]["01/bona_fide"], 16)
        self.assertEqual(r1["dev"]["availability"]["03/attack"], 22)
        self.assertEqual(doc["frames"]["distinct_frames_used"], len(self.fx.train) + len(self.fx.test))
        self.assertEqual(doc["frames"]["cache_frames"], len(self.fx.cache_keys))
        self.assertEqual(doc["frames"]["distinct_frames_to_extract"], len(self.fx.train) + len(self.fx.test) - len(self.fx.cache_keys))
        # manifest riservati: SHA-256 dichiarati nel pubblico, in_cache coerente, disgiunzione dei ruoli, righe ordinate
        for i in range(1, 5):
            keys = []
            for role in NF.ROLES:
                text = (run / "restricted" / f"fold{i}_{role}.csv").read_text()
                self.assertEqual(sha(text.encode()), doc["per_fold"][str(i)]["roles"][role]["manifest_sha256"])
                rows = list(csv.DictReader(io.StringIO(text)))
                self.assertTrue(all((r["key"] in self.fx.cache_keys) == (r["in_cache"] == "1") for r in rows))
                self.assertEqual([r["key"] for r in rows], sorted(r["key"] for r in rows))
                keys += [r["key"] for r in rows]
            self.assertEqual(len(keys), len(set(keys)))
        members = list(csv.DictReader(io.StringIO((run / "restricted" / "fold_members.csv").read_text())))
        self.assertEqual({(int(m["fold"]), m["subject"]) for m in members}, {(i, s) for i, v in FOLDS.items() for s in v})
        self.assertEqual(sha((run / "restricted" / "fold_members.csv").read_bytes()), doc["folds"]["members_sha256"])
        # Markdown: righe «available» e «to extract» uguali al JSON
        md = (run / "public" / "fold_schema_public.md").read_text()
        av = r1["fitting"]["availability"]
        self.assertIn("| fitting, available | " + " | ".join(str(av[f"{s}/{NF.CLASS_NAME[l]}"]) for s in NF.SESSIONS for l in NF.LABELS)
                      + f" | {sum(av.values())} |", md)
        to_extract = [sum(r1[role]["counts"][x]["to_extract"] for role in NF.ROLES)
                      for x in [f"{s}/{NF.CLASS_NAME[l]}" for s in NF.SESSIONS for l in NF.LABELS] + ["total"]]
        self.assertIn("| to extract (not in cache) | " + " | ".join(str(v) for v in to_extract) + " |", md)

    def test_public_has_no_identifier(self):
        run = self.run_ok()
        for name in ("fold_schema_public.json", "fold_schema_public.md"):
            text = (run / "public" / name).read_text()
            for s in SUBJECTS:
                self.assertNotIn(f"{s}_{s}_", text)
                self.assertNotIn(f'"{s}"', text)
            self.assertNotIn(".jpg", text)
            self.assertNotIn("real/", text)

    def test_sha256sums_complete_and_consistent(self):
        run = self.run_ok()
        for sub in ("public", "restricted"):
            listed = {}
            for line in (run / sub / "SHA256SUMS").read_text().splitlines():
                h, name = line.split("  ")
                listed[name] = h
                self.assertEqual(sha((run / sub / name).read_bytes()), h, name)
            self.assertEqual(sorted(listed), sorted(set(os.listdir(run / sub)) - {"SHA256SUMS"}))
        anchor = (run / "restricted" / "public_SHA256SUMS.sha256").read_text().split()[0]
        self.assertEqual(anchor, sha((run / "public" / "SHA256SUMS").read_bytes()))

    def test_no_old_pass_after_fail(self):
        run = self.run_ok()
        self.fx.cfg["sampling"]["fitting_per_cell"] = 50
        self.fx.write_config()
        code, out, err = self.fx.run("run")
        self.assertEqual(code, 1)
        self.assertIn("frames available, 50 required", err)
        self.assertEqual(json.loads((run / "status.json").read_text())["status"], "FAIL")
        self.assertEqual(os.listdir(run / "public"), [])
        self.assertEqual(os.listdir(run / "restricted"), [])

    def test_no_old_pass_after_manifest_failure(self):
        run = self.run_ok()
        self.fx.cfg["manifests"]["cnn_es"]["sha256"] = "0" * 64
        self.fx.write_config()
        code, out, err = self.fx.run("run")
        self.assertEqual(code, 1)
        self.assertIn("SHA-256", err)
        self.assertEqual(json.loads((run / "status.json").read_text())["status"], "FAIL")
        self.assertEqual(os.listdir(run / "public"), [])
        self.assertEqual(os.listdir(run / "restricted"), [])

    def test_foreign_file_in_run_folder_refused(self):
        run = self.fx.root / "results" / "folds" / RUN_ID
        (run / "public").mkdir(parents=True)
        (run / "public" / "other.txt").write_text("x")
        code, out, err = self.fx.run("run")
        self.assertEqual(code, 1)
        self.assertIn("not an output of this command", err)
        self.assertEqual((run / "public" / "other.txt").read_text(), "x")
        self.assertFalse((run / "status.json").exists())

    def test_protected_out_root_refused(self):
        (self.fx.root / "results" / "m2").mkdir()
        code, out, err = self.fx.run("run", out_root=self.fx.root / "results" / "m2")
        self.assertEqual(code, 1)
        self.assertFalse((self.fx.root / "results" / "m2" / RUN_ID).exists())

    def test_symlinked_out_root_refused(self):
        target = self.fx.tmp / "elsewhere"
        target.mkdir()
        link = self.fx.root / "results" / "folds_link"
        link.symlink_to(target)
        code, out, err = self.fx.run("run", out_root="results/folds_link")
        self.assertEqual(code, 1)
        self.assertIn("symbolic link", err)
        self.assertEqual(os.listdir(target), [])

    def test_input_inside_run_folder_or_run_inside_manifests_refused(self):
        run = self.fx.root / "results" / "folds" / RUN_ID
        run.mkdir(parents=True)
        inner = run / "manifests"
        shutil.copytree(self.fx.mdir, inner)
        code, out, err = self.fx.run("run", manifests=inner.relative_to(self.fx.root))
        self.assertEqual(code, 1)
        self.assertIn("inside the run folder", err)
        self.assertFalse((run / "status.json").exists())
        code, out, err = self.fx.run("run", out_root=self.fx.mdir)
        self.assertEqual(code, 1)
        self.assertIn("inside the manifests folder", err)
        self.assertFalse((self.fx.mdir / RUN_ID).exists())

    def test_public_identifier_refused(self):
        for field, value, pattern in (("status", "see real/1101_1101_00_00_01_0.jpg", "an image key"),
                                      ("declarations", {"note": "subject 1103"}, "subject identifier 1103")):
            self.fx.cfg[field] = value
            self.fx.write_config()
            code, out, err = self.fx.run("run")
            self.assertEqual(code, 1)
            self.assertIn("public outputs refused", err)
            self.assertIn(pattern, err)
            run = self.fx.root / "results" / "folds" / RUN_ID
            self.assertEqual(json.loads((run / "status.json").read_text())["status"], "FAIL")
            self.assertFalse((run / "public").exists() and os.listdir(run / "public"))
            self.fx.cfg = Fixture.make_config(self.fx, FIT, DEV)

    def test_failure_after_public_written_leaves_nothing(self):
        real_write = NF.mw.write_all
        calls = []

        def failing_write(out, files, newline=None):
            calls.append(Path(out).name)
            if Path(out).name == NF.RESTRICTED:
                raise OSError("disk full (simulated)")
            return real_write(out, files, newline)

        with mock.patch.object(NF.mw, "write_all", failing_write):
            code, out, err = self.fx.run("run")
        self.assertEqual(code, 1)
        self.assertIn("disk full", err)
        self.assertIn(NF.PUBLIC, calls)
        run = self.fx.root / "results" / "folds" / RUN_ID
        self.assertEqual(json.loads((run / "status.json").read_text())["status"], "FAIL")
        self.assertEqual(os.listdir(run / "public"), [])
        self.assertEqual(os.listdir(run / "restricted"), [])

    def test_dirty_tree_gets_dev_suffix_and_flag(self):
        with mock_root(self.fx.root):
            NF.REPO_STATE = lambda r: {"commit": "abcdef0", "dirty": True}
            self.assertEqual(NF.run_id_for("20261103", NF.repo_state(self.fx.root)), RUN_ID + "-dev")
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = NF.main(self.fx.args("run"))
        self.assertEqual(code, 0, err.getvalue())
        doc = json.loads((self.fx.root / "results" / "folds" / (RUN_ID + "-dev") / "public" / "fold_schema_public.json").read_text())
        self.assertEqual(doc["dirty"], True)


class TestRealInputs(unittest.TestCase):
    def test_real_manifests_check_inputs(self):
        cfg = REAL_ROOT / NF.DEFAULT_CONFIG
        mdir = REAL_ROOT / "results" / "m2" / "manifests"
        if not cfg.is_file() or not mdir.is_dir():
            skip_missing(self, "real manifests")
        c, h = NF.load_config(cfg)
        mans = NF.load_manifests(c, mdir)
        schema = NF.build_schema(c, mans)
        self.assertEqual(sorted(len(v) for v in schema["folds"].values()), [4, 4, 4, 4])
        self.assertEqual(dict(sorted((s, sum(1 for v in schema["strata"].values() if v == s)) for s in NF.STRATA)), {"A": 3, "B": 6, "C": 6, "D": 1})
        for i in range(1, 5):
            subs = schema["folds"][i]
            self.assertGreaterEqual(sum(1 for s in subs if schema["props"][s]["bona_fide_03"]), 2)
            self.assertGreaterEqual(sum(1 for s in subs if schema["props"][s]["sessions_01_02"]), 2)
        tested = [r["key"] for i in range(1, 5) for r in schema["per_fold"][i]["manifests"]["test"]]
        self.assertEqual(len(tested), 12614)
        self.assertEqual(len(set(tested) - schema["cache"]), 6614)
        for i in range(1, 5):
            pf = schema["per_fold"][i]
            self.assertEqual(len(pf["manifests"]["fitting"]), 2100)
            self.assertEqual(len(pf["manifests"]["dev"]), 600)


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
