"""
Scrive la configurazione di scripts/pad_arch_bn_factorial.py (pad-arch-bn-factorial-config/1) dal repository: SHA-256
della configurazione ARCHRUN, dei due status.json ARCHRUN (e dei SHA256SUMS riservati che ancorano), di pad_arch_run.py
e del modulo di pre-elaborazione di Keras, più le scelte fisse di NOTE.md. L'elenco delle unità è un parametro, così la
stessa implementazione serve alla prova sul sottoinsieme e ai run completi:

  --scope pilot     p4, fold 1, seed 42, entrambe le reti (la prima unità di run_order.csv): la prova
  --scope p4-24     i 24 checkpoint a patience 4 (primo run completo, con le metà Monte Carlo)
  --scope p12-24    i 24 checkpoint a patience 12 (secondo run completo, senza metà Monte Carlo)
  --units p4:LivenessNet:1:42,...   un elenco esplicito (ambito "custom": --cap-minutes, --duplicate-check e --mc-runs obbligatori)

Legge solo file di testo del repository (configurazioni, status.json dei run genitori, codice); non apre nulla sotto una
cartella restricted/, results/archeval o un manifest di test dei fold. Scrive un file JSON.

  .venv/bin/python -I -B delivery/archbn2x2/make_config.py --scope pilot --note delivery/archbn2x2/NOTE.md
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
import pad_arch_bn_factorial as F  # noqa: E402

PARENT_IDS = {"nuaa_patience_4": "20261120-ARCHRUN-p4-7faa49f", "nuaa_patience_12": "20261120-ARCHRUN-p12-ea7fdca"}
TAGS = {"p4": "nuaa_patience_4", "p12": "nuaa_patience_12"}
SEEDS, FOLDS = (42, 43, 44), (1, 2, 3, 4)
# valori per ambito: tetto in minuti di tempo reale (circa il doppio della stima della misura sintetica,
# pad_arch_bn_factorial.py measure), controllo dei duplicati, run con le metà Monte Carlo, metriche delle celle pubbliche
SCOPES = {
    "pilot": {"cap": 200, "duplicate_check": "all", "mc": ["nuaa_patience_4"], "report": True,
              "rule": "the first unit of plan/run_order.csv (fold 1, seed 42) at patience 4, one checkpoint per network: a neutral rule fixed "
                      "before any outcome, not chosen on results; the trial is followed by the public registration of code, configuration "
                      "and measured time, then by the complete runs p4-24 and p12-24 on fitting/dev, then by the test"},
    "p4-24": {"cap": 1500, "duplicate_check": "first", "mc": ["nuaa_patience_4"], "report": True,
              "rule": "the 24 checkpoints at patience 4 (the primary mechanistic subset of PROTOCOL.md §4), one process, with the Monte "
                      "Carlo halves"},
    "p12-24": {"cap": 820, "duplicate_check": "first", "mc": [], "report": True,
               "rule": "the 24 checkpoints at patience 12 (descriptive in PROTOCOL.md §4), one process after p4-24, without Monte Carlo halves"},
}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def units_of(scope: str, spec: str = None) -> list:
    if spec:
        out = []
        for part in spec.split(","):
            tag, net, fold, seed = part.strip().split(":")
            out.append({"run": TAGS[tag], "network": net, "fold": int(fold), "seed": int(seed)})
        return sorted(out, key=lambda u: u["fold"])
    if scope == "pilot":
        return [{"run": "nuaa_patience_4", "network": n, "fold": 1, "seed": 42} for n in F.NETWORKS]
    run = {"p4-24": "nuaa_patience_4", "p12-24": "nuaa_patience_12"}[scope]
    return [{"run": run, "network": n, "fold": f, "seed": s} for f in FOLDS for s in SEEDS for n in F.NETWORKS]


def scope_values(scope: str, args) -> dict:
    """I valori dell'ambito; per un elenco esplicito (custom) tetto, duplicati e Monte Carlo devono essere dati."""
    if scope == "custom":
        if args.cap_minutes is None or args.duplicate_check is None or args.mc_runs is None:
            raise SystemExit("--units needs --cap-minutes, --duplicate-check and --mc-runs (no value is taken from another scope)")
        return {"cap": args.cap_minutes, "duplicate_check": args.duplicate_check, "report": True,
                "mc": [r for r in args.mc_runs.split(",") if r], "rule": "an explicit list given to make_config.py"}
    d = dict(SCOPES[scope])
    if args.cap_minutes is not None:
        d["cap"] = args.cap_minutes
    if args.duplicate_check is not None:
        d["duplicate_check"] = args.duplicate_check
    if args.mc_runs is not None:
        d["mc"] = [r for r in args.mc_runs.split(",") if r]
    return d


def build(repo: Path, scope: str, units: list, args) -> dict:
    d = scope_values(scope, args)
    runcfg = repo / "delivery" / "archrun" / "pad_arch_run.json"
    parents = {}
    for r, rid in PARENT_IDS.items():
        folder = f"results/archrun/{rid}"
        b = (repo / folder / "status.json").read_bytes()
        st = json.loads(b)
        parents[r] = {"run_id": rid, "folder": folder, "status_sha256": sha(b), "restricted_sha256sums_sha256": st["restricted_sha256sums_sha256"]}
    note = None
    if args.note:
        p = repo / args.note
        note = sha(p.read_bytes()) if p.is_file() else None
    km = F.keras_module_path()
    return {
        "schema": F.CONFIG_SCHEMA,
        "task": "PROTOCOL.md §4 of the registered C2 follow-up (delivery/pad-arch-score-followup/PROTOCOL.md): the 2x2 re-estimation of the "
                "BatchNormalization moving statistics of saved ARCHRUN checkpoints, augmentation off/on x upstream dropout off/on during the "
                "re-estimation only, on the fitting frames of each unit, every cell scored on the development frames with augmentation and dropout "
                "off; fitting and development only, no test frame, no gradient step, no trained weight changed.",
        "authorisation": "The coordinator asked on 22 November 2026 to verify the 2x2 combination on fitting/dev with the saved checkpoints before any "
                         "new test score, and to register the code and the configuration after the trial with the measured time. --i-am-authorized "
                         "is passed only after NOTE.md has been frozen (analysis.sha256 set) in a private commit.",
        "operations": "launched from the repository root as nohup caffeinate -dims .venv/bin/python -I -B scripts/pad_arch_bn_factorial.py trial "
                      "--i-am-authorized > <a log file outside the repository> 2>&1 &, with a clean work tree and TMPDIR outside the repository; "
                      "SIGUSR1 stops it as INCOMPLETE at the next check, SIGTERM and SIGHUP as FAIL; one heavy process at a time on the machine; "
                      "results/ is versioned, so commit after the trial.",
        "scope": {"name": scope, "units": units, "rule": d["rule"], "report_cell_metrics": d["report"] if args.report is None else args.report == "yes"},
        "parent_config": {"path": "delivery/archrun/pad_arch_run.json", "sha256": sha(runcfg.read_bytes())},
        "parent_runs": parents,
        "code": {"scripts/pad_arch_run.py": sha((repo / "scripts" / "pad_arch_run.py").read_bytes())},
        "keras_functions": {"module": "keras.src.layers.preprocessing.image_preprocessing", "sha256": sha(km.read_bytes()),
                            "functions": ["transform", "get_rotation_matrix", "get_zoom_matrix", "get_translation_matrix"]},
        "recipe": {"source": "scripts/pad_arch_comparison.py", "function": "trial_trainer", "variable": "aug", "operations": F.recipe_json(F.RECIPE),
                   "applied_to": "the fitting frames as float32 divided by 255 once (as the training recipe and the estimator), before the network; "
                                 "the same operations in the same order, with the ranges derived from the recipe (recipe_ranges), fill mode reflect, "
                                 "bilinear interpolation, brightness clipped to [0, 1], contrast clipped to [0, 255] (its lower bound acts when the "
                                 "factor exceeds 1), noise not clipped",
                   "differences": "the random draws: per frame and view from a stateless counter-based NumPy Philox generator instead of the stateful "
                                  "Keras generators; the contrast factor is drawn per frame (Keras 2.15 draws one per batch of 16, whose composition was "
                                  "shuffled in training); same marginal distributions; the flip uses round(u) as the batched Keras flip"},
        "views": {"seeds": list(F.VIEW_SEEDS), "halves": [list(h) for h in F.HALVES], "order": "view outer, batch of 16 inner, frames in key order",
                  "storage": args.storage, "value_bounds": [-0.5, 1.7], "free_space_margin_gb": 2.0,
                  "folder": "with storage memmap the eight views of a fold (8 x frames x 256 x 256 x 3 float32, 8.8 GB for 1,400 frames) are "
                            "written to a new folder archbn2x2-views-* in the temporary folder of the process (TMPDIR), refused if inside the "
                            "repository or the input root, after a check that the free space covers the views plus the margin; the folder is "
                            "removed at the end of the fold and at every stop (signals included); a folder left by an earlier trial stops the "
                            "start; the views hold values derived from restricted frames and are never published"},
        "seeds": {"namespace": F.SEED_NAMESPACE, "domains": list(F.SEED_DOMAINS),
                  "derivation": "NumPy Philox4x64 keyed by the first 128 bits (big-endian) of SHA-256('pad-arch-bn-factorial/1|<domain>|<view "
                                "seed>|<SHA-256 hex of the frame key>|<operation name or Dropout layer name>'); one generator per draw; view "
                                "seeds 101-108 (view v uses seed 100 + v)",
                  "publication": "no per-frame seed, parameter, view or mask is written to any output (the frame key identifies subject and file)"},
        "dropout": {"networks": {"LivenessNet": {"layers": {"dropout1": 0.25, "dropout2": 0.25, "dropout_dense": 0.5}, "active": ["dropout1", "dropout2"]},
                                 "ResNet-18": {"layers": {}, "active": []}},
                    "mask_frequency_sigma": 5.0,
                    "rule": "only the Dropout layers before at least one BatchNormalization are active (dropout1 upstream of bn2_1, bn2_2 and "
                            "bn_dense; dropout2 upstream of bn_dense); dropout_dense follows the last BatchNormalization and is never active; every "
                            "layer, BatchNormalization included, is called with training=False; the masks are element-wise (noise_shape None is "
                            "checked); the GaussianNoise of the augmentation module is part of the augmentation factor, not of the dropout factor"},
        "cells": {c: F.cell_text(c) for c in F.CELLS},
        "reuse": {"duplicate_check": d["duplicate_check"],
                  "rule": "a layer whose active stochastic operations upstream are the same as in the reference cell (E01 -> E00, E11 -> E10) takes "
                          "the reference statistics (PROTOCOL.md §4: identical repeated views may be computed once; ResNet-18 duplicates may be "
                          "reused); duplicate_check all/first computes those layers explicitly as well, for every unit or for the first unit of "
                          "each network, and compares them within statistics_rel/abs (and the dev scores within 1e-5 when the whole cell is "
                          "reusable); a duplicate outside the tolerance stops the command (FAIL)"},
        "monte_carlo": {"runs": d["mc"],
                        "comparison": "two separate re-estimations per stochastic cell, views 101-104 and 105-108, each scored on the dev; effects "
                                      "per view half in effects.csv; montecarlo.csv: per run, network, metric and effect, H_u = effect(101-104) - "
                                      "effect(105-108), s_MC = sqrt(mean of H_u squared) / 2 without any reduction for the number of units, the "
                                      "mean H and |mean H| / 2, the ratio |effect| / s_MC and within_view_variability (|effect| <= 2 s_MC), for "
                                      "every unit and for the mean over the units (NOTE.md §6.4)"},
        "determinism": {"cell": "E11", "networks": ["LivenessNet"], "units": "first"},
        "tolerances": {"statistics_rel": 1e-4, "statistics_abs": 1e-6, "dev_probability_max_abs_diff": 1e-5, "dev_decisions_discordant": 0,
                       "consistency_rel": 1e-4, "consistency_abs": 1e-6, "softmax_of_logits_max_abs_diff": 1e-4},
        "threshold": 0.5,
        "estimator": {"batch": 16, "orders": F.PR.expected_orders(),
                      "accumulation": "nuaa_bn_recalibration.Moments: exact union of the batches in float64, view outer and batch inner, population "
                                      "variance; count = views x frames x H x W (frames for the dense layer); cast to float32"},
        "evaluation": {"dev": "nuaa_bn_recalibration.evaluate, unchanged: the clean dev frames / 255, batch 16, training=False (dropout and "
                              "augmentation off), probabilities and logits; decisions attack if score > 0.5 (a tie is bona fide)",
                       "fitting_clean": True},
        "threads": {"intra_op": 10, "inter_op": 10, "device": "CPU",
                    "rule": "pad_arch_comparison.set_threads right after the import of TensorFlow; the machine, versions and thread record must "
                            "equal those of the official trial (pad_arch_run.machine_rule, thread_environment_rule, threads_rule)"},
        "cap": {"wall_minutes": d["cap"], "unit": F.PAC.CAP_UNIT,
                "rule": "checked before every layer, pass, dev scoring and view batch; at the cap, or with SIGUSR1, INCOMPLETE: the complete units "
                        "are kept and reported, the unit in progress is not; set to about twice the estimate measured on synthetic arrays"},
        "memory": {"peak_rss_limit_gb": args.memory_limit_gb,
                   "rule": "the peak resident memory of the process (getrusage, which on macOS includes the memmap pages read) is checked at every "
                           "step; above the limit the command stops as FAIL with the value"},
        "ema": {"updates_per_epoch": 88, "fitting_frames": 1400, "train_batch": 16, "momentum": 0.99},
        "read_guard": {"forbidden_paths": ["results/archeval"], "forbidden_name_patterns": list(F.REQUIRED_PATTERNS),
                       "allowlisted_zones": list(F.GUARD_ZONES),
                       "outside_repository": "outside the repository and the input root only the Python installation, the system folders, the "
                                             "temporary folder, the Keras configuration folder, the folder of the configuration and (unofficial runs) "
                                             "the output root may be opened",
                       "rule": "an audit hook on every Python open (sys.addaudithook, event open) compares the opened file by identity (os.stat of "
                               "the file and of every folder that contains it; names compared in lower case) and refuses: the forbidden folders "
                               "of the repository and of the input root and any --forbid-outside folder; the forbidden names; any file with more "
                               "than one hard link outside the Python installation and the system folders; under results/ and data/ any file that "
                               "is not in a declared input or in the run folder; outside the repository and the input root anything not listed in "
                               "outside_repository. Reads done outside Python (TensorFlow's C++ code, git) are not audited"},
        "stop_rules": {
            "control_failure": "a failed control (topology, baseline, E00, E00 through the chain, tensors, consistency, counts, softmax, masks, "
                               "views, duplicates, determinism, EMA facts, weights, memory, disk space) stops the command as FAIL with failed_check "
                               "(check and values) in status.json; no stochastic cell is reported",
            "cap": "INCOMPLETE at the wall-time cap or with SIGUSR1 (exit code 2)",
            "signals": "SIGTERM and SIGHUP stop the command as FAIL; the views folder is removed",
            "memory": "FAIL above memory.peak_rss_limit_gb",
            "disk_space": "FAIL if the free space of the temporary folder is below the views plus views.free_space_margin_gb, or if the "
                          "temporary folder is inside the repository or the input root",
            "views": "FAIL if an augmented view holds a non-finite value or a value outside views.value_bounds",
            "masks": "FAIL if the zero frequency of a mask is outside mask_frequency_sigma binomial standard deviations of its rate",
            "determinism": "FAIL if the repeated cell is not bit-identical (statistics in float64 and float32, dev probabilities) or if the "
                           "regenerated first batch of a view differs from the stored one",
            "duplicates": "FAIL if an explicitly computed duplicate is outside the reference tolerance",
            "topology": "FAIL if LivenessNet is not the declared chain, if its Dropout layers, rates or noise shapes differ, or if ResNet-18 "
                        "holds a Dropout, noise or augmentation layer",
            "after_stop": "no automatic resume: a new attempt is a new process in a new folder (--attempt), reported with the earlier one"},
        "outputs": {"run": ["status.json", "run.log"], "public": list(F.PUBLIC_OUTPUTS),
                    "restricted": "<run>_<network>_fold<k>_seed<s>_<cell>_<views>_dev.csv and _bn.json per cell and view set; sealed_cells.csv, "
                                  "sealed_effects.csv and sealed_montecarlo.csv when the cell metrics are not public; SHA256SUMS"},
        "analysis": {"path": args.note or "delivery/archbn2x2/NOTE.md", "sha256": note,
                     "frozen_at": "NOTE.md (fixed choices; §6, the analysis proposed for the registration, frozen at the registration before any "
                                  "test score) is committed privately before the trial; trial refuses a null analysis.sha256"},
        "test": F.TEST_READ}


def main() -> int:
    ap = argparse.ArgumentParser(description="Writes the configuration of pad_arch_bn_factorial.py")
    ap.add_argument("--repo-root", default=str(F.ROOT))
    ap.add_argument("--scope", default="pilot", choices=sorted(SCOPES))
    ap.add_argument("--units", default=None, help="explicit units p4|p12:<network>:<fold>:<seed>, comma separated (scope custom)")
    ap.add_argument("--report", default=None, choices=["yes", "no"], help="cell metrics public (default: per scope)")
    ap.add_argument("--storage", default="memmap", choices=["memmap", "memory"])
    ap.add_argument("--memory-limit-gb", type=float, default=18.0)
    ap.add_argument("--cap-minutes", type=int, default=None)
    ap.add_argument("--duplicate-check", default=None, choices=["all", "first", "none"])
    ap.add_argument("--mc-runs", default=None, help="comma-separated runs with Monte Carlo halves (empty for none)")
    ap.add_argument("--note", default=None, help="NOTE.md relative to the repository root (its SHA-256 is pinned if the file exists)")
    ap.add_argument("--out", default=str(HERE.parent / "pad_arch_bn_factorial.json"))
    a = ap.parse_args()
    repo = Path(a.repo_root)
    scope = "custom" if a.units else a.scope
    doc = build(repo, scope, units_of(a.scope, a.units), a)
    text = json.dumps(doc, indent=1) + "\n"
    Path(a.out).write_text(text, encoding="utf-8")
    F.load_config(Path(a.out))           # la configurazione scritta passa la validazione dello script
    print(f"wrote {Path(a.out).name}: scope {scope}, {len(doc['scope']['units'])} units, SHA-256 {sha(text.encode())[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
