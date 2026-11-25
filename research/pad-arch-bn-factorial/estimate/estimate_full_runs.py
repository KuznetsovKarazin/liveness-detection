"""
Stima dei due run completi su fitting/dev (p4-24 e p12-24) dai tempi misurati nella prova pilot, con la stessa
composizione della stima sintetica di NOTE.md §8 (pad_arch_bn_factorial.estimate_scope ed estimate_unit): le viste di
ogni fold, più, per ogni unità, i passi che la configurazione dell'ambito prevede (le quattro celle sempre; i duplicati
espliciti solo sulla prima unità di ogni rete; le metà Monte Carlo solo a patience 4; la ripetizione di E11 solo sulla
prima unità di LivenessNet), come in pad_arch_bn_factorial.flags_for e in make_config.py.

Ingressi, solo file pubblici di questa cartella: runs/<prova>/public/timing.csv e runs/<prova>/status.json. Nessun
dato, nessun frame, nessun punteggio. Nessuna scrittura: stampa la stima.

Regola (scelta dell'autore della registrazione): una unità del run completo costa come l'unità della stessa rete nella
prova, meno gli intervalli misurati dei passi che il run completo non fa («(control)» dei duplicati espliciti, «(cell)»
delle sei o due metà Monte Carlo, «(cell)» della ripetizione); tutto il resto dell'unità (caricamento, baseline, E00,
copie dei modelli, confronti) resta. Le copie e i confronti dei passi tolti cadono fuori da quegli intervalli e restano
quindi nella stima, che per questo è leggermente per eccesso. Le viste costano per fold quanto il fold 1 della prova; il
tempo del processo fuori dalle unità e dalle viste (avvio e chiusura) è contato una volta per run.

  python -I -B estimate/estimate_full_runs.py          # dalla radice di questa cartella
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RUN = "20261124-ARCHBN2X2-pilot-6c4d6d8"
UNITS = {"LivenessNet": "p4_LivenessNet_fold1_seed42", "ResNet-18": "p4_ResNet-18_fold1_seed42"}
# i passi facoltativi di un'unità e le righe di timing.csv che li misurano (cella, strato)
STEPS = {"duplicates": [("E01 explicit", "(control)"), ("E11 explicit", "(control)")],
         "halves": [(f"{c} {h}", "(cell)") for c in ("E10", "E01", "E11") for h in ("101-104", "105-108")],
         "repeat": [("E11 repeat", "(cell)")]}
SYNTHETIC = {"pilot": 6009.0, "p4-24": 46716.0, "p12-24": 25534.0}     # NOTE.md §8
CAPS_MIN = {"pilot": 200, "p4-24": 1500, "p12-24": 820}                # NOTE.md §1
FOLDS, SEEDS = (1, 2, 3, 4), (42, 43, 44)
# come make_config.py: duplicate_check «all» nel pilot e «first» nei run completi; Monte Carlo solo a patience 4;
# determinismo: E11, LivenessNet, prima unità
SCOPES = {"pilot": {"units": [(1, 42)], "duplicates": "all", "halves": True},
          "p4-24": {"units": [(f, s) for f in FOLDS for s in SEEDS], "duplicates": "first", "halves": True},
          "p12-24": {"units": [(f, s) for f in FOLDS for s in SEEDS], "duplicates": "first", "halves": False}}


def load():
    rows = list(csv.DictReader((HERE / "runs" / RUN / "public" / "timing.csv").open(encoding="utf-8")))
    status = json.loads((HERE / "runs" / RUN / "status.json").read_text(encoding="utf-8"))
    return rows, status


def pick(rows, unit, cell, layer, col):
    got = [float(r[col]) for r in rows if r["unit"] == unit and r["cell"] == cell and r["layer"] == layer]
    if len(got) != 1:
        raise SystemExit(f"timing.csv: expected one row for {unit} / {cell} / {layer}, found {len(got)}")
    return got[0]


def measured(rows, status, col, status_key):
    """I tempi misurati della prova (colonna wall_seconds o process_cpu_seconds)."""
    m = {"views_per_fold": pick(rows, "fold1", "views", "(views)", col), "units": {}}
    for net, u in UNITS.items():
        steps = {k: sum(pick(rows, u, c, la, col) for c, la in v if any(r["unit"] == u and r["cell"] == c for r in rows))
                 for k, v in STEPS.items()}
        m["units"][net] = {"total": pick(rows, u, "(unit)", "(unit)", col), **steps}
    m["run_total"] = float(status[status_key])
    m["outside"] = m["run_total"] - m["views_per_fold"] - sum(x["total"] for x in m["units"].values())
    return m


def unit_cost(m, net, dup, halves, repeat):
    u = m["units"][net]
    return u["total"] - (0 if dup else u["duplicates"]) - (0 if halves else u["halves"]) - (0 if repeat else u["repeat"])


def estimate(m, scope):
    sc = SCOPES[scope]
    folds = {f for f, _ in sc["units"]}
    total = len(folds) * m["views_per_fold"] + m["outside"]
    detail = []
    for net in UNITS:
        for i, _ in enumerate(sc["units"]):
            first = i == 0
            dup = sc["duplicates"] == "all" or first
            rep = net == "LivenessNet" and first
            c = unit_cost(m, net, dup, sc["halves"], rep)
            detail.append((net, first, dup, sc["halves"], rep, c))
            total += c
    return total, detail, len(folds)


def main() -> int:
    rows, status = load()
    mw = measured(rows, status, "wall_seconds", "wall_seconds")
    mc = measured(rows, status, "process_cpu_seconds", "process_cpu_seconds")
    # le righe riassuntive di timing.csv (viste, celle, controlli, unità), nell'ordine del file
    print(f"rows of timing.csv of {RUN} for the views, each cell or control and each unit (wall s | CPU s):")
    for r in rows:
        if r["layer"] in ("(views)", "(re-estimation)", "(control)", "(cell)", "(unit)"):
            print(f"  {r['unit']} | {r['cell']} | {r['layer']} | {float(r['wall_seconds']):.1f} | {float(r['process_cpu_seconds']):.1f}")
    print()
    print(f"measured in {RUN} (timing.csv, status.json):")
    for lab, m in (("wall", mw), ("CPU", mc)):
        print(f"  {lab}: views of one fold {m['views_per_fold']:.1f} s; outside the units and the views {m['outside']:.1f} s; run total {m['run_total']:.1f} s")
        for net, u in m["units"].items():
            print(f"  {lab}: {net} unit {u['total']:.1f} s, of which explicit duplicates {u['duplicates']:.1f} s, "
                  f"Monte Carlo halves {u['halves']:.1f} s, repetition of E11 {u['repeat']:.1f} s")
    print()
    print("unit costs used (wall s): network, first unit or not, duplicates, halves, repetition")
    for scope in ("p4-24", "p12-24"):
        _, detail, _ = estimate(mw, scope)
        seen = {}
        for net, first, dup, hv, rep, c in detail:
            key = (net, first)
            seen.setdefault(key, [0, c, dup, hv, rep])[0] += 1
        for (net, first), (n, c, dup, hv, rep) in seen.items():
            print(f"  {scope}: {net}, {'first unit' if first else 'other units'} x {n}: duplicates {'yes' if dup else 'no'}, "
                  f"halves {'yes' if hv else 'no'}, repetition {'yes' if rep else 'no'}: {c:.1f} s each")
    print()
    print("scope | folds | wall s | wall h | wall min | cap min | wall / cap | CPU s | CPU h | synthetic estimate s (NOTE.md §8) | wall / synthetic")
    for scope in ("pilot", "p4-24", "p12-24"):
        w, _, nf = estimate(mw, scope)
        c, _, _ = estimate(mc, scope)
        print(f"{scope} | {nf} | {w:.1f} | {w / 3600:.2f} | {w / 60:.1f} | {CAPS_MIN[scope]} | {w / 60 / CAPS_MIN[scope]:.3f} | "
              f"{c:.1f} | {c / 3600:.2f} | {SYNTHETIC[scope]:.0f} | {w / SYNTHETIC[scope]:.3f}")
    w4, _, _ = estimate(mw, "p4-24")
    w12, _, _ = estimate(mw, "p12-24")
    print()
    print(f"check: the pilot recomposed from its parts is {estimate(mw, 'pilot')[0]:.1f} s against the measured {mw['run_total']:.1f} s")
    print(f"p4-24 + p12-24: {w4 + w12:.1f} s wall ({(w4 + w12) / 3600:.2f} h)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
