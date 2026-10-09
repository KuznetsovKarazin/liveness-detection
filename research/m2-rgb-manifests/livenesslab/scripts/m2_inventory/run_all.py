"""
Rigenera e stampa tutti i numeri della tabella 6.1 dell'inventario M2 di CASIA-FASD e CelebA-Spoof, confrontandoli con
i valori attesi dell'inventario dell'8/10/2026 (MATCH / DIFF per riga; codice di uscita 1 se qualcosa differisce).
Sola lettura: nessun file scritto, nessuna estrazione, nessun download, nessuna rete neurale caricata.
Dati attesi in data/ (vedi scripts/m2_inventory/common.py). Da lanciare dalla radice del repository:
    python scripts/m2_inventory/run_all.py [--private] [--json]
--private aggiunge gli identificativi per soggetto e per riga (solo per il riepilogo riservato, non per la scheda).
--json stampa anche i risultati completi dei quattro moduli.
I controlli su hard link e ore di modifica dipendono dalla copia locale: su una copia trasferita senza hard link o senza
mtime preservati risultano DIFF attesi. Il confronto del campo val_split usa `git show` sui commit 0ac7143 e 4d11e67:
serve un clone completo (non shallow) del repository. Date e ore sono calcolate nel fuso Europe/Rome.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import casia_inventory  # noqa: E402
import celeba_inventory  # noqa: E402
import checkpoints_inventory  # noqa: E402
import splits_inventory  # noqa: E402
from common import DATA, rel, strip_private  # noqa: E402

EXPECTED_SHA = {
    "AttackNetV1__casia_fasd": "c973c08970b36e74afc2f26cacf5045defe02cccb162f292697131266e7c8909",
    "AttackNetV2_1__casia_fasd": "c50be744cf393ca609446e72e139a01aca3ae7f1ceed3200e9e7a7d04f7ee59c",
    "AttackNetV2_2__casia_fasd": "1b3ba1700505ab14fb180115b0f91d11ecd8bfffc43f886c318f427c6d62366a",
    "LivenessNet__casia_fasd": "38152229e814638fa1dbe991fbd2138557a3b91bb465299059c7f3df5e88ca81",
    "AttackNetV2_2__casia_fasd-pooled": "4e6ab99ace1ed6a68dbc389ce0b6bd8406524a07f37f425e36b6d88fa58aeac0",
    "LivenessNet__casia_fasd-pooled": "c7cb5bf8aa8ca854fbe6d19edfb03371d555497983cf261f264f5ccd2e10dfc0",
    "AttackNetV1__celeba_spoof": "ddd4b19446621cd843714a7a87c7d9cf155498c94c88c80d0e123b4444e4cbcd",
    "AttackNetV2_1__celeba_spoof": "a78a8e964722a121eb5ee3680bd5827c54f2874cb637297df392e0148513e1ee",
    "AttackNetV2_2__celeba_spoof": "b9e4db69dd6873c1bb1c61d6db340a2749f8f9d20467769d64ae5ace42926e6a",
    "LivenessNet__celeba_spoof": "f2cac7ce396409049511be12cb2b855e647487966a06fdd7a30f49a0ba382ffd",
    "AttackNetV2_2__celeba_spoof-pooled": "fd813458862f8f187fb6b68a06cb2cd1d3780066f3cc31e094947c7a069b812a",
    "LivenessNet__celeba_spoof-pooled": "4f7553343edabc31d382a6bbd6fe1dff664b1727010737a2c94c89bf22bcc48d",
}


def rows(ca, ce, sp, ck):
    """(etichetta, valore calcolato, valore atteso) per ogni numero della tabella 6.1 e dei controlli citati nel testo."""
    a, n, pr = ca["archive"], ca["names"], ca["prepared"]
    f = ce["folders"]
    ru = ce["rule"]
    sc, sl = sp["casia_fasd"], sp["celeba_spoof"]
    cards = ck["cards"]
    off = lambda k: (cards[k]["n_train"], cards[k]["n_val"], cards[k]["n_test"])  # noqa: E731
    R = [
        # sorgenti
        ("CASIA revisione mirror", a["revision"], "4ea8ec45dee3765e17bb3efedfdcd7af39a3ebfe"),
        ("CASIA archivio byte", a["bytes"], 69265514),
        ("CASIA archivio SHA-256", a["sha256"], "76139d96360767adfecc1a8c01c062af14da664f665ba13ba5ce78111d0d9ec9"),
        ("CASIA SHA-256 = etag cache HF", a["sha256_equals_etag"], True),
        ("CASIA download", a["downloaded"][:19], "2026-09-12 11:39:21"),
        ("CASIA voci / file nell'archivio", (a["tar_entries"], a["tar_files"]), (8134, 8126)),
        ("CelebA revisione mirror", ce["revision"], "f158d9f71d786089d4f8b3b6c16b7bcae5989a3f"),
        ("CelebA parquet byte", ce["bytes"], 495744415),
        ("CelebA parquet SHA-256", ce["sha256"], "b5ac612899ebd7a825f86be4dbe3ffb84534b05bbc60172ba73e154b6e83cf09"),
        ("CelebA SHA-256 = etag cache HF", ce["sha256_equals_etag"], True),
        ("CelebA download", ce["downloaded"][:19], "2026-09-12 12:03:25"),
        ("CelebA colonne", ce["columns"], ["cropped_image", "labels", "labelNames"]),
        # unità e classi nel sorgente
        ("CASIA frame color train / test", (n["train"]["color_files"], n["test"]["color_files"]), (1655, 2408)),
        ("CASIA frame depth train / test", (n["train"]["depth_files"], n["test"]["depth_files"]), (1655, 2408)),
        ("CASIA train real / fake", (n["train"]["real"], n["train"]["fake"]), (404, 1251)),
        ("CASIA test real / fake", (n["test"]["real"], n["test"]["fake"]), (591, 1817)),
        ("CASIA soggetti train / test", (n["train"]["subjects"], n["test"]["subjects"]), (20, 30)),
        ("CASIA video train / test (12 per soggetto, nessuno mancante)",
         (n["train"]["videos"], n["test"]["videos"], n["train"]["missing_subject_video"] + n["test"]["missing_subject_video"]), (240, 360, 0)),
        ("CASIA frame per video train (min-max) / test (min-max)",
         (n["train"]["frames_per_video_min"], n["train"]["frames_per_video_max"], n["test"]["frames_per_video_min"], n["test"]["frames_per_video_max"]), (3, 15, 1, 19)),
        ("CASIA etichette coerenti con il token video", n["train"]["labels_consistent_with_video_token"] and n["test"]["labels_consistent_with_video_token"], True),
        ("CASIA frame 256x256 color / depth", (ca["images"]["sizes"].get("color:256x256"), ca["images"]["sizes"].get("depth:256x256")), (4063, 4063)),
        ("CASIA depth: file real, fake / fake con max < 32", (ca["images"]["depth_files"]["real"], ca["images"]["depth_files"]["fake"], ca["images"]["depth_max_below_32"]["fake"]), (995, 3068, 3068)),
        ("CelebA righe / con immagine", (ce["rows"], ce["rows"] - ce["rows_without_image"]), (6717, 6678)),
        ("CelebA live / spoof", (ce["labels"][0], ce["labels"][1]), (2030, 4687)),
        ("CelebA live / spoof con immagine", (ce["valid_rows_by_label"][0], ce["valid_rows_by_label"][1]), (2021, 4657)),
        ("CelebA righe senza immagine live / spoof", (ce["rows_without_image_by_label"].get(0, 0), ce["rows_without_image_by_label"].get(1, 0)), (9, 30)),
        ("CelebA path nullo / PNG", (ce["path_null"], ce["png_rows"]), (6717, 6678)),
        ("CelebA gruppi di duplicati esatti / cartelle", (ce["exact_duplicate_groups"], ce["duplicates_folders"]), (1, [["bonafide_training"]])),
        # cartelle preparate
        ("CASIA bonafide/attack_training", (pr["folders"]["bonafide_training"]["files"], pr["folders"]["attack_training"]["files"]), (404, 1251)),
        ("CASIA bonafide/attack_validation", (pr["folders"]["bonafide_validation"]["files"], pr["folders"]["attack_validation"]["files"]), (591, 1817)),
        ("CASIA cartelle = color del mirror, hard link", (all(v["equals_mirror_color"] for v in pr["folders"].values()), sum(v["hardlinked"] for v in pr["folders"].values())), (True, 4063)),
        ("CASIA eval real / attack", (pr["eval"]["real"]["files"], pr["eval"]["attack"]["files"]), (150, 150)),
        ("CASIA eval = spread(test,150), hard link", (pr["eval"]["real"]["equals_spread_test_150"] and pr["eval"]["attack"]["equals_spread_test_150"],
                                                      pr["eval"]["real"]["hardlinked"] + pr["eval"]["attack"]["hardlinked"]), (True, 300)),
        ("CASIA eval soggetti/video real, attack", (pr["eval"]["real"]["subjects"], pr["eval"]["real"]["videos"], pr["eval"]["attack"]["subjects"], pr["eval"]["attack"]["videos"]), (30, 89, 30, 150)),
        ("CelebA bonafide/attack_training", (f["bonafide_training"]["files"], f["attack_training"]["files"]), (1500, 1500)),
        ("CelebA bonafide/attack_validation", (f["bonafide_validation"]["files"], f["attack_validation"]["files"]), (450, 450)),
        ("CelebA eval real / attack", (ru["bonafide"]["eval_files"], ru["attack"]["eval_files"]), (150, 150)),
        ("CelebA regola riproduce le cartelle", all(ru[c][k] for c in ru for k in ("eval_equals_spread_valid_150", "training_equals_first_1500_rest", "validation_equals_eval_plus_next_300")), True),
        ("CelebA righe inutilizzate live / spoof", (ru["bonafide"]["unused_rows"], ru["attack"]["unused_rows"]), (71, 2707)),
        ("CelebA file byte-identici al parquet (images + eval)", sum(v["byte_identical_to_parquet"] for v in f.values()) + ru["bonafide"]["eval_byte_identical"] + ru["attack"]["eval_byte_identical"], 4200),
        ("CelebA etichette/dimensioni/sovrapposizioni errate", sum(v["label_mismatch"] + v["size_mismatch"] + v["null_rows"] for v in f.values()) + ce["folder_overlaps"], 0),
        # cache
        ("CASIA cache SHA-256", sc["cache"]["sha256"], "7d0a2b22146375bc4e629a430f0f5a161655d56ca26bb62a5e160cbff58bf5a6"),
        ("CelebA cache SHA-256", sl["cache"]["sha256"], "a085fea43bec3a130329645a52d6444feb2fe78e2d9e27d31fde2d9f8491f823"),
        ("CASIA cache train (real, attack) / test (real, attack)", (sc["cache"]["y_train"][0], sc["cache"]["y_train"][1], sc["cache"]["y_test"][0], sc["cache"]["y_test"][1]), (404, 1251, 591, 1500)),
        ("CelebA cache train (real, attack) / test (real, attack)", (sl["cache"]["y_train"][0], sl["cache"]["y_train"][1], sl["cache"]["y_test"][0], sl["cache"]["y_test"][1]), (1500, 1500, 450, 450)),
        ("CASIA cache = cartelle (y, s)", all(sc["cache_matches_folders"].values()), True),
        ("CelebA cache = cartelle (y, s = nome file)", all(sl["cache_matches_folders"].values()), True),
        ("CASIA s_train / s_test distinti", (sc["cache"]["s_train_distinct"], sc["cache"]["s_test_distinct"]), (20, 30)),
        ("CelebA s_train / s_test distinti", (sl["cache"]["s_train_distinct"], sl["cache"]["s_test_distinct"]), (3000, 900)),
        # split
        ("CASIA official ricostruito n_train/n_val/n_test", (sc["official_group_split"]["n_train"], sc["official_group_split"]["n_val"], sc["official_group_split"]["n_test"]), off("LivenessNet__casia_fasd")),
        ("CASIA official n_train/n_val/n_test (schede)", off("LivenessNet__casia_fasd"), (1339, 316, 2091)),
        ("CASIA early stopping: soggetti / video / soggetti condivisi con il fit", (sc["official_group_split"]["val_subjects"], sc["official_group_split"]["val_videos"], sc["official_group_split"]["fit_val_shared_subjects"]), (4, 48, 0)),
        ("CelebA official GroupShuffleSplit n_train/n_val (live, spoof)", (sl["official_group_split"]["n_train"], sl["official_group_split"]["n_val"], sl["official_group_split"]["val_by_class"][0], sl["official_group_split"]["val_by_class"][1]), (2400, 600, 295, 305)),
        ("CelebA official stratificato n_train/n_val (live, spoof)", (sl["official_stratified_alt"]["n_train"], sl["official_stratified_alt"]["n_val"], sl["official_stratified_alt"]["val_by_class"][0], sl["official_stratified_alt"]["val_by_class"][1]), (2400, 600, 300, 300)),
        ("CelebA official n_train/n_val/n_test (schede)", off("LivenessNet__celeba_spoof"), (2400, 600, 900)),
        ("CASIA pooled ricostruito", (sc["pooled"]["n_train"], sc["pooled"]["n_val"], sc["pooled"]["n_test"]), off("LivenessNet__casia_fasd-pooled")),
        ("CASIA pooled (schede)", off("LivenessNet__casia_fasd-pooled"), (2696, 300, 750)),
        ("CelebA pooled ricostruito", (sl["pooled"]["n_train"], sl["pooled"]["n_val"], sl["pooled"]["n_test"]), off("LivenessNet__celeba_spoof-pooled")),
        ("CelebA pooled (schede)", off("LivenessNet__celeba_spoof-pooled"), (2808, 312, 780)),
        ("CASIA pooled test real / attack", (sc["pooled"]["test_by_class"][0], sc["pooled"]["test_by_class"][1]), (199, 551)),
        ("CelebA pooled test real / attack", (sl["pooled"]["test_by_class"][0], sl["pooled"]["test_by_class"][1]), (390, 390)),
        ("CASIA eval nel training pooled", (sc["pooled"]["eval_in_pooled_trainval"], sc["pooled"]["eval_files"]), (235, 300)),
        ("CelebA eval nel training pooled", (sl["pooled"]["eval_in_pooled_trainval"], sl["pooled"]["eval_files"]), (244, 300)),
        ("CASIA pooled: video / soggetti su entrambi i lati", (sc["pooled"]["videos_both_sides"], sc["pooled"]["videos_total"], sc["pooled"]["subjects_both_sides"], sc["pooled"]["subjects_total"]), (444, 600, 50, 50)),
        # checkpoint
        ("Checkpoint CASIA / CelebA (schede)", (sum(k.split("__")[1].startswith("casia") for k in cards), sum(k.split("__")[1].startswith("celeba") for k in cards)), (6, 6)),
        ("SHA .h5 = scheda = manifest.json", sum(v["card_sha_ok"] and v["manifest_sha_ok"] for v in cards.values()), 12),
        ("SHA .h5 = valori attesi", sum(cards[k]["h5_sha256"] == s for k, s in EXPECTED_SHA.items()), 12),
        (".keras omonimi presenti", sum(v["keras_exists"] for v in cards.values()), 12),
        ("val_split CelebA official in 0ac7143 / 4d11e67 / oggi",
         sorted({(v["val_split_history"]["0ac7143"], v["val_split_history"]["4d11e67"], v["val_split"]) for k, v in cards.items() if k.endswith("__celeba_spoof")}),
         [("per soggetto", "casuale per immagine", "casuale per immagine")]),
        ("Cache scritte alle (CASIA, CelebA) / primo 'saved'", (ck["cache_mtime"]["casia_fasd_1500.npz"][:16], ck["cache_mtime"]["celeba_spoof_1500.npz"][:16], ck["first_saved"]),
         ("2026-09-12 12:20", "2026-09-12 12:23", "2026-09-12 12:58")),
        ("File versionati in models/weights oltre alle schede", sorted(Path(t).name for t in ck["tracked_other"]), ["PROVENANCE.md", "manifest.json"]),
    ]
    return R


def norm(v):
    return json.loads(json.dumps(v, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    missing = [rel(p) for p in (DATA / "raw" / "casiafasd.tar.gz", DATA / "raw" / "celeba_spoof", DATA / "CASIA_FASD" / "images",
                                DATA / "CelebA_Spoof" / "images", DATA / "processed") if not p.exists()]
    if missing:
        print("ERRORE: dati non trovati (da lanciare dalla radice del repository con i dati in data/): " + ", ".join(missing))
        sys.exit(2)
    ca = casia_inventory.run()
    ce = celeba_inventory.run()
    sp = splits_inventory.run()
    ck = checkpoints_inventory.run()
    bad = 0
    for label, got, exp in rows(ca, ce, sp, ck):
        ok = norm(got) == norm(exp)
        bad += not ok
        print(f"{'MATCH' if ok else 'DIFF '}  {label}: {norm(got)}" + ("" if ok else f"   (atteso {norm(exp)})"))
    for k, v in ck["cards"].items():
        print(f"       {k}: {v['protocol']} {v['n_train']}/{v['n_val']}/{v['n_test']} test {v['test_real']}/{v['test_attack']} sha {v['h5_sha256'][:8]}…{v['h5_sha256'][-4:]}")
    if a.private:
        print("RISERVATO soggetti di early stopping CASIA:", sp["casia_fasd"]["official_group_split"]["_val_subject_ids"])
        print("RISERVATO righe duplicate CelebA:", ce["_duplicate_rows"])
        print("RISERVATO intervalli di righe CelebA:", {c: (ce["rule"][c]["_training_row_range"], ce["rule"][c]["_validation_row_range"]) for c in ce["rule"]})
    if a.json:
        allr = {"casia": ca, "celeba": ce, "splits": sp, "checkpoints": ck}
        print(json.dumps(allr if a.private else strip_private(allr), indent=1, default=str))
    print(f"{'TUTTI I NUMERI COINCIDONO' if not bad else f'{bad} DIFFERENZE'}")
    sys.exit(1 if bad else 0)
