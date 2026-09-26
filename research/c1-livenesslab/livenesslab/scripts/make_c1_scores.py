"""
File dei punteggi per campione nel formato del gruppo di ricerca (research/templates/score-schema.json), a partire
dall'export di scripts/eval_dataset.py e dalla configurazione C1:
  results/c1/<run-id>/scores.csv   una riga per (immagine, analizzatore) con run_id, sample_id, dataset_id, split,
                                   subject_id, session_id, video_id, attack_type, modality, label, score_attack,
                                   threshold, prediction, checkpoint_sha256, config_sha256 e, in più, analyzer_id
  results/c1/<run-id>/SHA256SUMS   hash di scores.csv, manifest e configurazione
Il file contiene record per soggetto: va nello spazio riservato (Drive 02_Experiments/C1/<run-id>), non nel repository.
Uso:  python scripts/make_c1_scores.py [--dataset nuaa] [--run-id 20260926-C1-seed42-abcdef1]
"""
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUAA_NAME = re.compile(r"^(\d{4})_\d{4}_(\d{2})_")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="nuaa"); ap.add_argument("--run-id", default=None)
    ap.add_argument("--c1", default=str(ROOT / "results" / "c1"))
    a = ap.parse_args()
    c1 = Path(a.c1)
    cfg_path = c1 / f"{a.dataset}_config.json"; cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    short = (cfg["repository"]["commit"] or subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip())[:7]
    run_id = a.run_id or f"{time.strftime('%Y%m%d')}-C1-seed42-{short}"
    thr = float(cfg["conventions"]["threshold"])
    ck = {}
    for c in cfg["checkpoints"]:
        if c.get("weights_file"):
            ck[Path(c["weights_file"]).stem] = c["sha256"]
    fp_of = {an["id"]: an["fingerprint"] for an in cfg["analyzers"]}
    out = c1 / run_id; out.mkdir(parents=True, exist_ok=True)
    cfg_sha = sha256(cfg_path); man_sha = cfg["dataset"]["manifest_sha256"]
    n = 0
    with (c1 / f"{a.dataset}_scores.csv").open(encoding="utf-8") as f, (out / "scores.csv").open("w", newline="", encoding="utf-8") as g:
        w = csv.writer(g, lineterminator="\n")     # LF: gli hash restano validi anche dopo la normalizzazione di Git
        w.writerow(["run_id", "sample_id", "dataset_id", "split", "subject_id", "session_id", "video_id", "attack_type", "modality", "label",
                    "score_attack", "threshold", "prediction", "checkpoint_sha256", "config_sha256", "analyzer_id"])
        for r in csv.DictReader(f):
            if r["error"] == "1" or r["attack_score"] in ("", "nan"):
                continue
            name = Path(r["file"]).name; m = NUAA_NAME.match(name)
            score = float(r["attack_score"]); label = int(r["label"])
            fp = r["fingerprint"]
            # impronta "<id>:<sha16>" delle CNN -> hash completo del checkpoint dalla configurazione; altrimenti l'impronta stessa
            sha = next((ck[k] for k in ck if k.lower().replace("__", "__") and ck[k].startswith(fp.split(":")[-1][:16])), fp)
            w.writerow([run_id, r["file"], a.dataset, "test", m.group(1) if m else "", m.group(2) if m else "", "",
                        "print" if label == 1 else "", "RGB", label, f"{score:.6f}", thr, 1 if score > thr else 0, sha, cfg_sha, r["analyzer"]])
            n += 1
    sums = out / "SHA256SUMS"
    sums.write_text("".join(f"{sha256(out / 'scores.csv')}  scores.csv\n{man_sha}  {cfg['dataset']['manifest']}\n{cfg_sha}  {cfg_path.name}\n"), encoding="utf-8")
    print(f"run-id {run_id}: {n} record in {out / 'scores.csv'}; hash in {sums}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
