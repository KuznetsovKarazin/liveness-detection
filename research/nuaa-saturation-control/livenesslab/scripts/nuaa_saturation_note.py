"""
Early dev saturation of the NUAA LivenessNet trainings: the facts behind the one-page note that proposes ONE control
(recalibration of the BatchNormalization statistics of the patience-4 checkpoints on the fitting set, no training).

Reads only public tables already merged in research/pad-2026 and the public architecture file; no training, no
inference, no model loaded, no image read, no test score or test metric read. Every source is hashed (SHA-256) and the
hashes are written to tables/sources.json; `check` rebuilds everything in a temporary folder, compares byte for byte
and reports any source whose hash differs from the recorded one.

Sources (paths relative to the research/ folder, plus src/architectures.py next to it):
  nuaa-degenerate-trainings/tables/models.csv        the 36 models of the patience-4 run (rows run = sessablrun)
  nuaa-degenerate-trainings/tables/curves.csv        their per-epoch train/dev histories
  nuaa-degenerate-trainings/tables/aggregate.json    criteria (degenerate, saturation, recovery), BN momentum
  nuaa-session-ablation-run/tables/run_record.json   run id, architecture hash, per-model timings (threshold step)
  nuaa-session-ablation-run/tables/aggregate.json    only the "training" block (weights_sha256 of each checkpoint)
  nuaa-patience-verification/tables/run_record.json  run id and architecture hash of the patience-12 run
  nuaa-patience-verification/comparison/curves.csv   per-epoch curves of the patience-12 run
  nuaa-patience-verification/comparison/comparison.csv
                                                     only the parent_* columns (dev AUROC at the patience-4 checkpoint)
  nuaa-constant-lr-control/tables/run_record.json    run id and architecture hash of the constant-lr run
  nuaa-constant-lr-control/comparison/curves.csv     per-epoch curves of the constant-lr run
  nuaa-constant-lr-control/livenesslab/delivery/constlr/nuaa_session_ablation_run_constant_lr.json
                                                     fitting and dev sizes, batch, scoring device
  ../src/architectures.py                            LivenessNet layer order (parsed as text)

Commands:
  build --research research --out research/nuaa-saturation-control/tables
  check --research research --out research/nuaa-saturation-control/tables [--note research/nuaa-saturation-control/NOTE.md]
"""
import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import sys
import tempfile
from pathlib import Path

SOURCES = {
    "models": "nuaa-degenerate-trainings/tables/models.csv",
    "curves": "nuaa-degenerate-trainings/tables/curves.csv",
    "degenerate_aggregate": "nuaa-degenerate-trainings/tables/aggregate.json",
    "p4_record": "nuaa-session-ablation-run/tables/run_record.json",
    "p4_aggregate": "nuaa-session-ablation-run/tables/aggregate.json",
    "p12_record": "nuaa-patience-verification/tables/run_record.json",
    "p12_curves": "nuaa-patience-verification/comparison/curves.csv",
    "p12_comparison": "nuaa-patience-verification/comparison/comparison.csv",
    "p12c_record": "nuaa-constant-lr-control/tables/run_record.json",
    "p12c_curves": "nuaa-constant-lr-control/comparison/curves.csv",
    "config": "nuaa-constant-lr-control/livenesslab/delivery/constlr/nuaa_session_ablation_run_constant_lr.json",
    "architecture": "../src/architectures.py",
}
ARMS = ("arm_b", "arm_a_star", "arm_b_prime")
ARM_LABEL = {"arm_b": "B", "arm_a_star": "A*", "arm_b_prime": "B'"}
FOLDS = (1, 2, 3, 4)
SEEDS = (42, 43, 44)
RECALIBRATION_PASSES = 5          # una passata sul fitting per ogni strato BatchNormalization, in ordine
FITTING_EVAL_PASSES = 2           # misura secondaria: fitting in inferenza prima e dopo la ricalibrazione
DEV_PASSES = 2                    # dev prima e dopo la ricalibrazione
LIVENESSNET_LAYERS = 22
CHECKPOINT_COLUMNS = ("arm", "fold", "seed", "n_fitting", "steps_per_epoch", "epochs_run", "restored_epoch",
                      "val_acc_at_best", "best_val_loss", "degenerate", "train_accuracy_epoch1",
                      "train_accuracy_restored", "dev_accuracy_restored", "epochs_at_half", "bn_residual_epoch1",
                      "bn_residual_restored", "first_recovery_epoch", "at_criterion_restored",
                      "dev_auroc_checkpoint", "weights_sha256")
OUTPUTS = ("checkpoints.csv", "facts.json", "facts.md", "layers.md", "sources.json")


class NoteError(Exception):
    pass


def fail(msg: str) -> None:
    raise NoteError(msg)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_source(research: Path, key: str) -> bytes:
    """Byte di una sorgente; errore esplicito se manca o non è un file regolare."""
    p = research / SOURCES[key]
    if not p.is_file():
        fail("source missing: %s (%s)" % (SOURCES[key], key))
    return p.read_bytes()


def parse_csv(raw: bytes, name: str, required: tuple) -> list:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    rows = []
    for r in reader:
        if None in r or any(v is None for v in r.values()):
            fail("%s: line %d has %s fields than the header" % (name, reader.line_num, "more" if None in r else "fewer"))
        rows.append(r)
    if not rows:
        fail("%s: no rows" % name)
    missing = [c for c in required if c not in rows[0]]
    if missing:
        fail("%s: missing columns %s" % (name, ", ".join(missing)))
    return rows


def parse_json(raw: bytes, name: str) -> dict:
    try:
        d = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        fail("%s: not valid JSON (%s)" % (name, e))
    if not isinstance(d, dict):
        fail("%s: a JSON object is expected" % name)
    return d


def num(s, name: str) -> float:
    try:
        v = float(s)
    except (TypeError, ValueError):
        fail("%s: %r is not a number" % (name, s))
    if not math.isfinite(v):
        fail("%s: %r is not finite" % (name, s))
    return v


def is_half(v: float, tol: float) -> bool:
    return abs(v - 0.5) <= tol


def residual(momentum: float, steps: int) -> float:
    """Quota della media mobile ancora dovuta al valore iniziale dopo `steps` aggiornamenti: momentum ** steps."""
    if not (0.0 < momentum < 1.0) or steps < 0:
        fail("residual: momentum must be in (0, 1) and steps >= 0, got %r and %r" % (momentum, steps))
    return momentum ** steps


def steps_per_epoch(n: int, batch: int) -> int:
    if n <= 0 or batch <= 0:
        fail("steps per epoch: n and batch must be positive, got %r and %r" % (n, batch))
    return -(-n // batch)


def f4(v: float) -> str:
    return "%.4f" % v


def thousands(n: int) -> str:
    return "{:,}".format(n)


def med_range(values: list) -> dict:
    if not values:
        fail("median of an empty list")
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "n": len(values)}


# ---------------------------------------------------------------------------------------------------------------------
# architettura

LAYER_RE = re.compile(r'layers\.(\w+)\((.*)name="(\w+)"\)')


def parse_livenessnet(text: str) -> list:
    """Strati di LivenessNet nell'ordine del sorgente: [(posizione, tipo, nome)] dal blocco della classe."""
    m = re.search(r"^class LivenessNet\b.*?(?=^class |\Z)", text, re.S | re.M)
    if not m:
        fail("architecture: class LivenessNet not found")
    block = m.group(0)
    if "momentum" in block:
        fail("architecture: LivenessNet sets a BatchNormalization momentum; the note assumes the Keras default")
    layers = []
    for i, line in enumerate(block.splitlines(), 1):
        if not re.search(r"layers\.\w+\(", line):
            continue
        lm = LAYER_RE.search(line)
        if not lm:
            fail("architecture: line %d of LivenessNet creates a layer without name=\"...\"" % i)
        layers.append((len(layers) + 1, lm.group(1), lm.group(3)))
    if len(layers) != LIVENESSNET_LAYERS:
        fail("architecture: LivenessNet should have %d layers, found %d" % (LIVENESSNET_LAYERS, len(layers)))
    return layers


def layer_facts(layers: list) -> dict:
    bn = [(p, n) for p, t, n in layers if t == "BatchNormalization"]
    dr = [(p, n) for p, t, n in layers if t == "Dropout"]
    if len(bn) != 5 or len(dr) != 3:
        fail("architecture: LivenessNet should have 5 BatchNormalization and 3 Dropout layers, found %d and %d"
             % (len(bn), len(dr)))
    # per ogni BN: il dropout più vicino che lo precede nella rete (se c'è) e se è il primo BN dopo quel dropout
    after, first = {}, {}
    for p, n in bn:
        prev = [d for d in dr if d[0] < p]
        after[n] = prev[-1][1] if prev else None
        first[n] = bool(prev) and not any(prev[-1][0] < q < p for q, _ in bn)
    return {"batch_normalization": [{"position": p, "name": n, "preceded_by_dropout": after[n],
                                     "first_after_dropout": first[n]} for p, n in bn],
            "dropout": [{"position": p, "name": n} for p, n in dr],
            "layers": len(layers)}


def layers_md(layers: list) -> str:
    out = ["# LivenessNet layer order (parsed from src/architectures.py)", "",
           "| position | type | name | mark |", "|---|---|---|---|"]
    for p, t, n in layers:
        mark = "BN: recalibrated" if t == "BatchNormalization" else ("dropout: off in inference" if t == "Dropout" else "")
        out.append("| %d | %s | %s | %s |" % (p, t, n, mark))
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------------------
# costruzione

def by_model(rows: list, name: str) -> dict:
    """{(arm, fold, seed): {epoch: row}} con controllo di epoche contigue da 1."""
    out = {}
    for r in rows:
        k = (r["arm"], int(r["fold"]), int(r["seed"]))
        e = int(r["epoch"])
        if e in out.setdefault(k, {}):
            fail("%s: duplicated epoch %d for %s fold %d seed %d" % (name, e, *k))
        out[k][e] = r
    for k, h in out.items():
        if sorted(h) != list(range(1, len(h) + 1)):
            fail("%s: epochs of %s fold %d seed %d are not 1..n" % (name, *k))
    return out


def expected_models() -> list:
    return [(a, f, s) for a in ARMS for f in FOLDS for s in SEEDS]


def half_epochs(h: dict, tol: float) -> list:
    return [e for e in sorted(h) if is_half(num(h[e]["val_accuracy"], "val_accuracy"), tol)]


def build(research: Path) -> dict:
    raw = {k: read_source(research, k) for k in SOURCES}
    sources = {SOURCES[k].replace("../", ""): sha256_bytes(v) for k, v in raw.items()}

    agg = parse_json(raw["degenerate_aggregate"], SOURCES["degenerate_aggregate"])
    crit = agg["criteria"]
    tol = float(crit["degenerate"]["tolerance"])
    if float(crit["degenerate"]["val_acc_at_best"]) != 0.5 or sorted(crit["degenerate"]["restored_epochs"]) != [1, 2]:
        fail("degenerate criterion differs from restored epoch 1-2 with dev accuracy 0.5")
    rec_acc, rec_loss = float(crit["recovery"]["val_accuracy_min"]), float(crit["recovery"]["val_loss_max"])
    momentum = float(agg["recipe"]["bn_momentum"])

    rec4 = parse_json(raw["p4_record"], SOURCES["p4_record"])
    rec12 = parse_json(raw["p12_record"], SOURCES["p12_record"])
    rec12c = parse_json(raw["p12c_record"], SOURCES["p12c_record"])
    arch_sha = sources["src/architectures.py"]
    for name, r in (("p4", rec4), ("p12", rec12), ("p12c", rec12c)):
        if r.get("status") != "PASS" or r.get("dirty") is not False:
            fail("%s run record: status PASS from a clean tree expected" % name)
        if r["code_sha256"]["architecture"] != arch_sha:
            fail("%s run record: architecture SHA-256 %s differs from src/architectures.py %s"
                 % (name, r["code_sha256"]["architecture"], arch_sha))
    if agg["inputs"]["ablation_run"]["run_id"] != rec4["run_id"]:
        fail("the degenerate study does not read the patience-4 run of the run record")

    cfg = parse_json(raw["config"], SOURCES["config"])
    if cfg["architecture"]["sha256"] != arch_sha:
        fail("configuration: architecture SHA-256 differs from src/architectures.py")
    batch = int(cfg["training"]["batch"])
    n_fit = {a: int(cfg["arms"][a]["total"]) for a in ARMS}
    n_dev = int(cfg["dev"]["total"])
    score_batch, device = int(cfg["scoring"]["batch"]), cfg["scoring"]["device"]
    if device != "CPU" or cfg["training"]["device"] != "CPU":
        fail("configuration: the training and scoring device must be CPU")

    layers = parse_livenessnet(raw["architecture"].decode("utf-8"))
    lf = layer_facts(layers)

    models = [r for r in parse_csv(raw["models"], SOURCES["models"],
                                   ("run", "arm", "fold", "seed", "n_fitting", "steps_per_epoch", "epochs_run",
                                    "restored_epoch", "best_val_loss", "val_acc_at_best", "degenerate",
                                    "first_recovery_epoch", "epochs_with_val_acc_0.5", "train_seconds", "seconds_per_epoch"))
              if r["run"] == "sessablrun"]
    mk = {(r["arm"], int(r["fold"]), int(r["seed"])): r for r in models}
    if sorted(mk) != sorted(expected_models()) or len(models) != 36:
        fail("models.csv: the 36 models of the patience-4 run (3 arms x 4 folds x 3 seeds) expected, found %d" % len(models))
    curves = by_model([r for r in parse_csv(raw["curves"], SOURCES["curves"],
                                            ("run", "arm", "fold", "seed", "epoch", "accuracy", "val_loss", "val_accuracy"))
                       if r["run"] == "sessablrun"], "curves.csv")
    if sorted(curves) != sorted(expected_models()):
        fail("curves.csv: curves of the 36 models expected")
    training = parse_json(raw["p4_aggregate"], SOURCES["p4_aggregate"])["training"]
    auroc = {}
    for r in parse_csv(raw["p12_comparison"], SOURCES["p12_comparison"],
                       ("arm", "fold", "seed", "parent_restored_epoch", "parent_dev_auroc_at_checkpoint")):
        auroc[(r["arm"], int(r["fold"]), int(r["seed"]))] = r
    timings = rec4["timings"]

    rows = []
    for k in expected_models():
        arm, fold, seed = k
        m, h = mk[k], curves[k]
        name = "%s_fold%d_seed%d" % k
        nf, spe, ep_run, rest = int(m["n_fitting"]), int(m["steps_per_epoch"]), int(m["epochs_run"]), int(m["restored_epoch"])
        if nf != n_fit[arm]:
            fail("%s: n_fitting %d differs from the configuration (%d)" % (name, nf, n_fit[arm]))
        if spe != steps_per_epoch(nf, batch):
            fail("%s: %d steps per epoch, ceil(%d / %d) = %d expected" % (name, spe, nf, batch, steps_per_epoch(nf, batch)))
        if len(h) != ep_run or not 1 <= rest <= ep_run:
            fail("%s: %d epochs in curves.csv, epochs_run %d, restored epoch %d" % (name, len(h), ep_run, rest))
        vl = [num(h[e]["val_loss"], name + " val_loss") for e in sorted(h)]
        if vl.index(min(vl)) + 1 != rest:
            fail("%s: the restored epoch %d is not the first epoch of minimum val_loss" % (name, rest))
        acc_rest = num(h[rest]["val_accuracy"], name + " val_accuracy")
        if abs(acc_rest - num(m["val_acc_at_best"], name)) > tol or abs(vl[rest - 1] - num(m["best_val_loss"], name)) > 1e-9:
            fail("%s: models.csv and curves.csv disagree at the restored epoch" % name)
        degen = rest in (1, 2) and is_half(acc_rest, tol)
        flag = m["degenerate"]
        if flag not in ("0", "1"):
            fail("%s: degenerate flag %r" % (name, flag))
        if (flag == "1") != degen:
            fail("%s: flagged degenerate=%s but the curves give restored epoch %d with dev accuracy %s"
                 % (name, flag, rest, h[rest]["val_accuracy"]))
        halves = half_epochs(h, tol)
        if len(halves) != int(m["epochs_with_val_acc_0.5"]):
            fail("%s: %d epochs at 0.5 in the curves, %s in models.csv" % (name, len(halves), m["epochs_with_val_acc_0.5"]))
        first_rec = next((e for e in sorted(h) if num(h[e]["val_accuracy"], name) >= rec_acc
                          and num(h[e]["val_loss"], name) < rec_loss), None)
        if (str(first_rec) if first_rec else "") != m["first_recovery_epoch"]:
            fail("%s: first recovery epoch %s from the curves, %r in models.csv" % (name, first_rec, m["first_recovery_epoch"]))
        tr = training.get(name)
        if not tr or (int(tr["restored_epoch"]), int(tr["n_fitting"]), int(tr["epochs_run"])) != (rest, nf, ep_run) \
                or abs(float(tr["best_val_loss"]) - vl[rest - 1]) > 1e-9:
            fail("%s: the training block of the patience-4 aggregate disagrees with models.csv" % name)
        if int(tr["n_dev"]) != n_dev:
            fail("%s: n_dev %s, configuration %d" % (name, tr["n_dev"], n_dev))
        wsha = str(tr["weights_sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", wsha):
            fail("%s: weights_sha256 is not a SHA-256" % name)
        spe_s = num(m["seconds_per_epoch"], name)
        if abs(spe_s - num(m["train_seconds"], name) / ep_run) > 0.01:
            fail("%s: seconds_per_epoch %s is not train_seconds / epochs_run" % (name, m["seconds_per_epoch"]))
        a = auroc.get(k)
        if not a or int(a["parent_restored_epoch"]) != rest:
            fail("%s: comparison.csv has no parent row with the same restored epoch" % name)
        if "threshold " + name not in timings:
            fail("%s: no threshold timing in the patience-4 run record" % name)
        rows.append({
            "arm": arm, "fold": fold, "seed": seed, "n_fitting": nf, "steps_per_epoch": spe, "epochs_run": ep_run,
            "restored_epoch": rest, "val_acc_at_best": acc_rest, "best_val_loss": vl[rest - 1], "degenerate": int(degen),
            "train_accuracy_epoch1": num(h[1]["accuracy"], name), "train_accuracy_restored": num(h[rest]["accuracy"], name),
            "dev_accuracy_restored": acc_rest, "epochs_at_half": len(halves),
            "bn_residual_epoch1": residual(momentum, spe), "bn_residual_restored": residual(momentum, spe * rest),
            "first_recovery_epoch": first_rec,
            "at_criterion_restored": int(acc_rest >= rec_acc and vl[rest - 1] < rec_loss),
            "dev_auroc_checkpoint": num(a["parent_dev_auroc_at_checkpoint"], name + " dev AUROC"),
            "weights_sha256": wsha, "seconds_per_epoch": num(m["seconds_per_epoch"], name),
            "threshold_seconds": float(timings["threshold " + name]["monotonic_seconds"]),
            "saturated_after_epoch2": len([e for e in halves if e >= 3]), "half_at_epoch1": int(1 in halves),
        })

    # curve delle due run a 12 epoche: epoche a 0,5 dei dieci e dei 26
    ten = [r for r in rows if r["degenerate"]]
    rest_rows = [r for r in rows if not r["degenerate"]]
    ten_keys = {(r["arm"], r["fold"], r["seed"]) for r in ten}
    spe_of = {(r["arm"], r["fold"], r["seed"]): r["steps_per_epoch"] for r in rows}
    later = {}
    for key, rk, run_id in (("p12", "p12_curves", rec12["run_id"]), ("p12c", "p12c_curves", rec12c["run_id"])):
        cur = by_model(parse_csv(raw[rk], SOURCES[rk], ("arm", "fold", "seed", "in_ten", "epoch", "val_accuracy")), SOURCES[rk])
        if sorted(cur) != sorted(expected_models()):
            fail("%s: curves of the 36 models expected" % SOURCES[rk])
        for k, h in cur.items():
            if ((h[1]["in_ten"] == "1") != (k in ten_keys)) or any(h[e]["in_ten"] != h[1]["in_ten"] for e in h):
                fail("%s: in_ten of %s fold %d seed %d differs from the degenerate models of the patience-4 run" % (SOURCES[rk], *k))
        t_half = {k: half_epochs(cur[k], tol) for k in sorted(ten_keys)}
        counts = [len(v) for v in t_half.values()]
        o_half = [len(half_epochs(cur[k], tol)) for k in cur if k not in ten_keys]
        last = max((max(v) for v in t_half.values() if v), default=0)
        # residuo al più tardi saturato, con i passi per epoca del modello (il massimo se più modelli condividono l'epoca)
        at_last = [k for k, v in t_half.items() if v and max(v) == last]
        res_last = max((residual(momentum, spe_of[k] * last) for k in at_last), default=None)
        later[key] = {"run_id": run_id, "epochs": max(len(h) for h in cur.values()),
                      "ten_with_half_epoch": sum(1 for c in counts if c > 0), "ten_half_epochs": med_range(counts),
                      "ten_epoch1_half": sum(1 for k in ten_keys if 1 in t_half[k]),
                      "ten_last_half_epoch": last, "ten_last_half_arms": sorted({ARM_LABEL[k[0]] for k in at_last}),
                      "residual_at_last_half_epoch": res_last,
                      "others_with_half_epoch": sum(1 for c in o_half if c > 0)}

    facts = make_facts(rows, ten, rest_rows, lf, later, momentum, tol, rec_acc, rec_loss, n_fit, n_dev, batch,
                       score_batch, device, rec4["run_id"], agg["run_id"],
                       [int(v) for v in agg["inputs"]["ablation_run"]["subjects_per_fold_dev"].values()])
    return {"rows": rows, "facts": facts, "layers": layers, "sources": sources}


def make_facts(rows, ten, rest_rows, lf, later, momentum, tol, rec_acc, rec_loss, n_fit, n_dev, batch, score_batch,
               device, run4, degen_run, dev_subjects) -> dict:
    if len(ten) + len(rest_rows) != 36:
        fail("36 models expected")
    spe = {a: steps_per_epoch(n_fit[a], batch) for a in ARMS}
    # costo: stima derivata (nessuna misura); limite prudente e stima centrale
    thr = [r["threshold_seconds"] for r in rows]
    thr_med, thr_max = statistics.median(thr), max(thr)
    spe_max = {a: max(r["seconds_per_epoch"] for r in rows if r["arm"] == a) for a in ARMS}
    per_frame = thr_med / (2 * n_dev)       # la fase della soglia fa due passate sul dev (probabilità e logit)
    cost = {}
    for a in ARMS:
        fit_passes = RECALIBRATION_PASSES + FITTING_EVAL_PASSES
        frames = fit_passes * n_fit[a] + DEV_PASSES * n_dev
        cost[a] = {"frames_per_model": frames,
                   "upper_seconds_per_model": fit_passes * spe_max[a] + thr_max,
                   "central_seconds_per_model": frames * per_frame + thr_med}
    n_arm = {a: sum(1 for r in rows if r["arm"] == a) for a in ARMS}
    total_up = sum(cost[a]["upper_seconds_per_model"] * n_arm[a] for a in ARMS)
    total_c = sum(cost[a]["central_seconds_per_model"] * n_arm[a] for a in ARMS)
    restored = {}
    for r in ten:
        restored[str(r["restored_epoch"])] = restored.get(str(r["restored_epoch"]), 0) + 1
    tr1_ten = med_range([r["train_accuracy_epoch1"] for r in ten])
    tr1_rest = med_range([r["train_accuracy_epoch1"] for r in rest_rows])
    au = sorted(r["dev_auroc_checkpoint"] for r in ten)
    return {
        "runs": {"patience_4": run4, "patience_12": later["p12"]["run_id"], "patience_12_constant_lr": later["p12c"]["run_id"],
                 "degenerate_study": degen_run},
        "criteria": {"degenerate": "restored epoch 1 or 2 with dev accuracy 0.5 (within %g)" % tol,
                     "recovery": "dev accuracy >= %g and val_loss < %g" % (rec_acc, rec_loss)},
        "criteria_values": {"val_accuracy_min": rec_acc, "val_loss_max": rec_loss, "tolerance": tol},
        "models": {"total": len(rows), "degenerate": len(ten), "non_degenerate": len(rest_rows),
                   "degenerate_by_arm": {ARM_LABEL[a]: sum(1 for r in ten if r["arm"] == a) for a in ARMS},
                   "degenerate_restored_epochs": restored,
                   "degenerate_dev_accuracy_half_at_restored": sum(1 for r in ten if is_half(r["dev_accuracy_restored"], tol)),
                   "degenerate_best_val_loss": med_range([r["best_val_loss"] for r in ten]),
                   "non_degenerate_at_criterion_at_checkpoint": sum(r["at_criterion_restored"] for r in rest_rows),
                   "weights_sha256_listed": sum(1 for r in rows if r["weights_sha256"]),
                   "weights_sha256_source": SOURCES["p4_aggregate"] + " (block training, key weights_sha256)"},
        "train_accuracy_epoch1": {"degenerate": tr1_ten, "non_degenerate": tr1_rest},
        "train_accuracy_restored_degenerate": med_range([r["train_accuracy_restored"] for r in ten]),
        "dev_saturation": {"degenerate_at_epoch1": sum(r["half_at_epoch1"] for r in ten),
                           "non_degenerate_at_epoch1": sum(r["half_at_epoch1"] for r in rest_rows),
                           "non_degenerate_at_epoch3_or_later": sum(1 for r in rest_rows if r["saturated_after_epoch2"] > 0)},
        "dev_auroc_checkpoint_degenerate": {"values": au, "above_0.9": sum(1 for v in au if v > 0.9),
                                            "below_0.5": sum(1 for v in au if v < 0.5), "equal_0.5": sum(1 for v in au if v == 0.5)},
        "bn": {"momentum": momentum,
               "residual_after_epoch": {ARM_LABEL[a]: [residual(momentum, spe[a] * e) for e in (1, 2, 3)] for a in ARMS},
               "residual_restored_degenerate": med_range([r["bn_residual_restored"] for r in ten])},
        "layers": lf,
        "sizes": {"fitting": {ARM_LABEL[a]: n_fit[a] for a in ARMS}, "steps_per_epoch": {ARM_LABEL[a]: spe[a] for a in ARMS},
                  "dev": n_dev, "batch": batch, "scoring_batch": score_batch, "device": device},
        "cost": {"derived_not_measured": True, "passes_on_fitting": RECALIBRATION_PASSES, "fitting_evaluation_passes": FITTING_EVAL_PASSES, "dev_passes": DEV_PASSES,
                 "threshold_step_seconds": {"median": thr_med, "max": thr_max, "dev_passes_in_step": 2},
                 "slowest_per_epoch_mean": {ARM_LABEL[a]: spe_max[a] for a in ARMS},
                 "per_arm": {ARM_LABEL[a]: cost[a] for a in ARMS}, "models_per_arm": {ARM_LABEL[a]: n_arm[a] for a in ARMS},
                 "total_upper_minutes": total_up / 60.0, "total_central_minutes": total_c / 60.0},
        "later_runs": later,
        "dev_subjects_per_fold": {"min": min(dev_subjects), "max": max(dev_subjects)},
        "non_degenerate_outside_criterion": [
            {"arm": ARM_LABEL[r["arm"]], "fold": r["fold"], "seed": r["seed"], "restored_epoch": r["restored_epoch"],
             "dev_accuracy": r["dev_accuracy_restored"], "best_val_loss": r["best_val_loss"], "dev_auroc": r["dev_auroc_checkpoint"]}
            for r in rest_rows if not r["at_criterion_restored"]],
    }


# ---------------------------------------------------------------------------------------------------------------------
# uscite

def fmt_cell(c: str, v) -> str:
    if v is None:
        return "NA"
    if c in ("val_acc_at_best", "best_val_loss", "train_accuracy_epoch1", "train_accuracy_restored", "dev_accuracy_restored",
             "dev_auroc_checkpoint"):
        return repr(float(v))
    if c in ("bn_residual_epoch1", "bn_residual_restored"):
        return "%.6f" % v
    return str(v)


def checkpoints_csv(rows: list) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CHECKPOINT_COLUMNS)
    for r in rows:
        w.writerow([fmt_cell(c, r[c]) for c in CHECKPOINT_COLUMNS])
    return buf.getvalue()


def rounded(o):
    if isinstance(o, float):
        return round(o, 6)
    if isinstance(o, dict):
        return {k: rounded(v) for k, v in o.items()}
    if isinstance(o, list):
        return [rounded(v) for v in o]
    return o


WORDS = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def word(n: int) -> str:
    if n not in WORDS:
        fail("no word for %r" % n)
    return WORDS[n]


def note_numbers(f: dict) -> list:
    """Frammenti che NOTE.md deve contenere alla lettera (spazi normalizzati), formattati dai fatti ricostruiti."""
    m, c, s = f["models"], f["cost"], f["sizes"]
    t1d, t1n = f["train_accuracy_epoch1"]["degenerate"], f["train_accuracy_epoch1"]["non_degenerate"]
    res = f["bn"]["residual_after_epoch"]
    p12, p12c = f["later_runs"]["p12"], f["later_runs"]["p12c"]
    au = f["dev_auroc_checkpoint_degenerate"]
    ds = f["dev_saturation"]
    first = [b for b in f["layers"]["batch_normalization"] if b["first_after_dropout"]]
    pc = c["per_arm"]
    ten, rest = m["degenerate"], m["non_degenerate"]
    out = [
        "run `%s`" % f["runs"]["patience_4"],
        "momentum %g" % f["bn"]["momentum"],
        "%d steps per epoch for B and A\\*, %d for B'" % (s["steps_per_epoch"]["B"], s["steps_per_epoch"]["B'"]),
        "`%s` and `%s` (first after `%s`, `%s`)" % (first[0]["name"], first[1]["name"], first[0]["preceded_by_dropout"],
                                                   first[1]["preceded_by_dropout"]),
        "hypotheses 4 and 7",
        "dev accuracy 0.5 at epoch 1 in %d of %d and %d of %d" % (ds["degenerate_at_epoch1"], ten, ds["non_degenerate_at_epoch1"], rest),
        "%s [%s-%s] (the %s), %s (the %d)" % (f4(t1d["median"]), f4(t1d["min"]), f4(t1d["max"]), word(ten), f4(t1n["median"]), rest),
        "residual (weight of the initial value) %s after epoch 1 (B, A\\*; B' %s)" % (f4(res["B"][0]), f4(res["B'"][0])),
        "residual %s after epoch 3 (B, A\\*; B' %s)" % (f4(res["B"][2]), f4(res["B'"][2])),
        "%d of the %d hit 0.5 at epoch 3 or later" % (ds["non_degenerate_at_epoch3_or_later"], rest),
        "%d-%d epochs at 0.5 (%d-%d, constant lr)" % (p12["ten_half_epochs"]["min"], p12["ten_half_epochs"]["max"],
                                                     p12c["ten_half_epochs"]["min"], p12c["ten_half_epochs"]["max"]),
        "epoch %d (%s, residual %s)" % (p12["ten_last_half_epoch"], ", ".join(p12["ten_last_half_arms"]),
                                       f4(p12["residual_at_last_half_epoch"])),
        "%s of the %s have dev AUROC 0.5000" % (word(au["equal_0.5"]), word(ten)),
        "The %s tied at AUROC 0.5000" % word(au["equal_0.5"]),
        "the %s that already order the dev (AUROC %s, %s)" % (word(au["above_0.9"]), f4(au["values"][-1]), f4(au["values"][-2])),
        "the %s, saturated at epoch %s; the %d as controls" % (word(ten), " or ".join(sorted(m["degenerate_restored_epochs"])), rest),
        "the %s BatchNormalization layers" % word(len(f["layers"]["batch_normalization"])),
        "(%s frames for B and A\\*, %s for B')" % (thousands(s["fitting"]["B"]), thousands(s["fitting"]["B'"])),
        "(%d frames, batch %d)" % (s["dev"], s["scoring_batch"]),
        "dev accuracy >= %r, val_loss < %r" % (f["criteria_values"]["val_accuracy_min"], f["criteria_values"]["val_loss_max"]),
        "residual ≤ %s" % f4(res["B"][2]),
        "the %d of the %d meeting it" % (m["non_degenerate_at_criterion_at_checkpoint"], rest),
        "%s recalibration and %s evaluation passes over the fitting set, %s over the dev" % (
            word(c["passes_on_fitting"]), word(c["fitting_evaluation_passes"]), word(c["dev_passes"])),
        "(%s frame passes for B and A\\*, %s for B')" % (thousands(pc["B"]["frames_per_model"]), thousands(pc["B'"]["frames_per_model"])),
        "about %.0f minutes for %d models, upper estimate about %.0f minutes" % (c["total_central_minutes"], m["total"], c["total_upper_minutes"]),
        "Small dev (%d-%d subjects)" % (f["dev_subjects_per_fold"]["min"], f["dev_subjects_per_fold"]["max"]),
    ]
    for o in f["non_degenerate_outside_criterion"]:
        out.append("%s fold %d seed %d (restored epoch %d, dev accuracy %s, not degenerate)" % (o["arm"], o["fold"], o["seed"],
                                                                                           o["restored_epoch"], f4(o["dev_accuracy"])))
    return out


def facts_md(f: dict) -> str:
    m, s, c = f["models"], f["sizes"], f["cost"]
    t1d, t1n = f["train_accuracy_epoch1"]["degenerate"], f["train_accuracy_epoch1"]["non_degenerate"]
    trr = f["train_accuracy_restored_degenerate"]
    res = f["bn"]["residual_after_epoch"]
    rr = f["bn"]["residual_restored_degenerate"]
    bl = m["degenerate_best_val_loss"]
    ds = f["dev_saturation"]
    au = f["dev_auroc_checkpoint_degenerate"]
    L = ["# Facts for the saturation control note (generated; every number from the public sources in `sources.json`)", "",
         "Runs: patience 4 `%s`, patience 12 `%s`, patience 12 at a constant learning rate `%s`; degenerate study `%s`."
         % (f["runs"]["patience_4"], f["runs"]["patience_12"], f["runs"]["patience_12_constant_lr"], f["runs"]["degenerate_study"]), "",
         "| fact | value |", "|---|---|",
         "| models of the patience-4 run | %d |" % m["total"],
         "| degenerate (%s) | %d of %d (B %d, A* %d, B' %d) |" % (f["criteria"]["degenerate"], m["degenerate"], m["total"],
                                                              m["degenerate_by_arm"]["B"], m["degenerate_by_arm"]["A*"], m["degenerate_by_arm"]["B'"]),
         "| restored epoch of degenerate | %s |" % ", ".join("epoch %s: %d" % (k, v) for k, v in sorted(m["degenerate_restored_epochs"].items())),
         "| dev accuracy 0.5 at the restored epoch, degenerate | %d of %d |" % (m["degenerate_dev_accuracy_half_at_restored"], m["degenerate"]),
         "| best val_loss of degenerate | median %s [%s-%s] |" % (f4(bl["median"]), f4(bl["min"]), f4(bl["max"])),
         "| train accuracy at epoch 1, degenerate | median %s [%s-%s] |" % (f4(t1d["median"]), f4(t1d["min"]), f4(t1d["max"])),
         "| train accuracy at epoch 1, non-degenerate | median %s [%s-%s] |" % (f4(t1n["median"]), f4(t1n["min"]), f4(t1n["max"])),
         "| train accuracy at the restored epoch, degenerate | median %s [%s-%s] |" % (f4(trr["median"]), f4(trr["min"]), f4(trr["max"])),
         "| dev AUROC at the checkpoint, degenerate | %s (above 0.9: %d; exactly 0.5: %d; below 0.5: %d) |"
         % (", ".join(f4(v) for v in au["values"]), au["above_0.9"], au["equal_0.5"], au["below_0.5"]),
         "| dev accuracy 0.5 at epoch 1 | degenerate %d of %d, non-degenerate %d of %d |" % (ds["degenerate_at_epoch1"], m["degenerate"], ds["non_degenerate_at_epoch1"], m["non_degenerate"]),
         "| non-degenerate at the recovery criterion at their checkpoint | %d of %d |" % (m["non_degenerate_at_criterion_at_checkpoint"], m["non_degenerate"]),
         "| non-degenerate with a dev accuracy of 0.5 at epoch 3 or later | %d of %d |" % (f["dev_saturation"]["non_degenerate_at_epoch3_or_later"], m["non_degenerate"]),
         "| BN momentum (Keras default, no momentum argument in LivenessNet) | %s |" % f["bn"]["momentum"],
         ]
    for a in ("B", "A*", "B'"):
        L.append("| BN residual after epochs 1, 2, 3, %s (%d steps per epoch) | %s |"
                 % (a, s["steps_per_epoch"][a], ", ".join(f4(v) for v in res[a])))
    L += ["| BN residual at the restored epoch, degenerate | median %s [%s-%s] |" % (f4(rr["median"]), f4(rr["min"]), f4(rr["max"])),
          "| BatchNormalization layers (position) | %s |" % ", ".join("%s (%d)" % (b["name"], b["position"]) for b in f["layers"]["batch_normalization"]),
          "| first BN layer after each dropout | %s |" % ", ".join("%s after %s" % (b["name"], b["preceded_by_dropout"])
                                                            for b in f["layers"]["batch_normalization"] if b["first_after_dropout"]),
          "| BN layers downstream of a dropout | %s |" % ", ".join(b["name"] for b in f["layers"]["batch_normalization"] if b["preceded_by_dropout"]),
          "| Dropout layers (position) | %s |" % ", ".join("%s (%d)" % (d["name"], d["position"]) for d in f["layers"]["dropout"]),
          "| fitting frames per model | B %s, A* %s, B' %s |" % (thousands(s["fitting"]["B"]), thousands(s["fitting"]["A*"]), thousands(s["fitting"]["B'"])),
          "| dev frames per fold; batch; scoring batch; device | %d; %d; %d; %s |" % (s["dev"], s["batch"], s["scoring_batch"], s["device"])]
    for key, label in (("p12", "patience 12"), ("p12c", "patience 12, constant lr")):
        p = f["later_runs"][key]
        L.append("| %s: degenerate with an epoch at 0.5; epochs at 0.5 per model; at epoch 1; last epoch at 0.5 (arm, BN residual) | %d of %d; %d-%d (median %s); %d of %d; epoch %d (%s, %s) |"
                 % (label, p["ten_with_half_epoch"], m["degenerate"], p["ten_half_epochs"]["min"], p["ten_half_epochs"]["max"],
                    "%g" % p["ten_half_epochs"]["median"], p["ten_epoch1_half"], m["degenerate"], p["ten_last_half_epoch"],
                    ", ".join(p["ten_last_half_arms"]), f4(p["residual_at_last_half_epoch"])))
    L += ["| dev subjects per fold | %d-%d |" % (f["dev_subjects_per_fold"]["min"], f["dev_subjects_per_fold"]["max"])]
    for o in f["non_degenerate_outside_criterion"]:
        L.append("| non-degenerate outside the recovery criterion at its checkpoint | %s fold %d seed %d: restored epoch %d, dev accuracy %s, val_loss %s, dev AUROC %s |"
                 % (o["arm"], o["fold"], o["seed"], o["restored_epoch"], f4(o["dev_accuracy"]), f4(o["best_val_loss"]), f4(o["dev_auroc"])))
    L += ["| cost inputs (patience-4 run record) | threshold step (load + 2 dev passes of %d frames, CPU) median %.1f s, max %.1f s; slowest per-epoch mean of the arm (train_seconds / epochs_run, maximum over its models) B %.1f s, A* %.1f s, B' %.1f s |"
          % (s["dev"], c["threshold_step_seconds"]["median"], c["threshold_step_seconds"]["max"],
             c["slowest_per_epoch_mean"]["B"], c["slowest_per_epoch_mean"]["A*"], c["slowest_per_epoch_mean"]["B'"]),
          "| passes per model | %d recalibration and %d evaluation passes over the fitting set, %d over the dev |"
          % (c["passes_on_fitting"], c["fitting_evaluation_passes"], c["dev_passes"])]
    for a in ("B", "A*", "B'"):
        p = c["per_arm"][a]
        L.append("| cost per model, %s (%s frame passes) | central %.0f s, upper %.0f s |" % (a, thousands(p["frames_per_model"]),
                                                                                         p["central_seconds_per_model"], p["upper_seconds_per_model"]))
    L += ["| cost, 36 models (derived, not measured) | central %.0f minutes, upper %.0f minutes |" % (c["total_central_minutes"], c["total_upper_minutes"]),
          "", "Central: frame passes x (median threshold step / %d frame passes) + median threshold step. Upper: the %d fitting "
          "passes each priced as the slowest per-epoch mean of the arm + the slowest threshold step (which holds the two dev "
          "passes). Neither prices the reading of the fitting frames." % (2 * s["dev"], c["passes_on_fitting"] + c["fitting_evaluation_passes"]), ""]
    return "\n".join(L)


def outputs(doc: dict) -> dict:
    f = doc["facts"]
    return {
        "checkpoints.csv": checkpoints_csv(doc["rows"]),
        "facts.json": json.dumps(rounded(f), indent=1, sort_keys=True) + "\n",
        "facts.md": facts_md(f),
        "layers.md": layers_md(doc["layers"]),
        "sources.json": json.dumps({"sha256": doc["sources"], "note": "SHA-256 of every source read at build time"},
                                   indent=1, sort_keys=True) + "\n",
    }


def write_outputs(files: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        with open(out / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)


# ---------------------------------------------------------------------------------------------------------------------
# comandi

def cmd_build(a) -> int:
    doc = build(Path(a.research))
    write_outputs(outputs(doc), Path(a.out))
    f = doc["facts"]
    print("BUILD: %d checkpoints, %d degenerate, %d sources -> %s" % (len(doc["rows"]), f["models"]["degenerate"],
                                                                     len(doc["sources"]), a.out))
    print("RESULT: PASS")
    return 0


def cmd_check(a) -> int:
    out = Path(a.out)
    bad = []
    rec_path = out / "sources.json"
    if not rec_path.is_file():
        fail("%s missing: run build first" % rec_path)
    recorded = json.loads(rec_path.read_text(encoding="utf-8"))["sha256"]
    research = Path(a.research)
    for key, rel in SOURCES.items():
        p = research / rel
        shown = rel.replace("../", "")
        if not p.is_file():
            bad.append("%s: source missing" % shown)
        elif recorded.get(shown) != sha256_bytes(p.read_bytes()):
            bad.append("%s: SHA-256 differs from the one recorded in sources.json" % shown)
    if not bad:
        doc = build(research)
        with tempfile.TemporaryDirectory() as tmp:
            write_outputs(outputs(doc), Path(tmp))
            for name in OUTPUTS:
                p = out / name
                if not p.is_file():
                    bad.append("%s: missing" % name)
                elif p.read_bytes() != (Path(tmp) / name).read_bytes():
                    bad.append("%s: differs from the rebuilt file (byte comparison)" % name)
        if a.note:
            text = " ".join(Path(a.note).read_text(encoding="utf-8").split())     # a capo e spazi normalizzati
            for s in note_numbers(doc["facts"]):
                if s not in text:
                    bad.append("%s: %r not found" % (Path(a.note).name, s))
    for b in bad:
        print("PROBLEM: " + b)
    print("CHECK: %d files%s, %d problems" % (len(OUTPUTS), " + note" if a.note else "", len(bad)))
    print("RESULT: %s" % ("PASS" if not bad else "FAIL"))
    return 0 if not bad else 1


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, text in (("build", "write the tables"), ("check", "rebuild in a temporary folder and compare byte for byte")):
        s = sub.add_parser(name, help=text)
        s.add_argument("--research", required=True, help="the research/ folder of the repository (src/ is next to it)")
        s.add_argument("--out", required=True, help="the tables folder")
    sub.choices["check"].add_argument("--note", default=None, help="NOTE.md: every number of a fixed list must appear in it")
    return p


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    try:
        return cmd_build(a) if a.cmd == "build" else cmd_check(a)
    except NoteError as e:
        print("ERROR: %s" % e)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, OSError, csv.Error) as e:
        print("ERROR: a source has an unexpected structure or cannot be read: %s: %s" % (type(e).__name__, e))
    print("RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
