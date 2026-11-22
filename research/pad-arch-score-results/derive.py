"""Summaries fixed in NOTES.md of the registration (points 3 and 4), derived from the aggregate outputs only.

Reads det_grid.csv and score_metrics.csv of a run and writes, in the output folder:
  det_mean.csv         unweighted arithmetic mean of minimum_attainable_test_bpcer over the twelve fold-seed units,
                       for each patience, network, BN state and APCER ceiling (96 rows);
  operating_points.csv for each target (5%, 10%) and each achieved development APCER, the number of checkpoint-states,
                       and how many of them are all-attack (development APCER 0 with development BPCER 1).
Standard library only. Usage: python derive.py <run>/public <output folder>
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main():
    source, out = Path(sys.argv[1]), Path(sys.argv[2])
    grid, metrics = read(source / "det_grid.csv"), read(source / "score_metrics.csv")
    if len(grid) != 1152 or len(metrics) != 96:
        raise SystemExit("expected 1,152 grid rows and 96 metric rows")
    groups = defaultdict(list)
    order = []
    for row in grid:
        key = (row["patience"], row["network"], row["bn_state"], row["test_apcer_ceiling"])
        if key not in groups:
            order.append(key)
        groups[key].append(float(row["minimum_attainable_test_bpcer"]))
    if len(order) != 96 or {len(v) for v in groups.values()} != {12}:
        raise SystemExit("expected 96 groups of twelve curves")
    write(out / "det_mean.csv",
          ["patience", "network", "bn_state", "test_apcer_ceiling", "units", "mean_minimum_attainable_test_bpcer"],
          [[*key, len(groups[key]), repr(sum(groups[key]) / len(groups[key]))] for key in order])
    points = defaultdict(lambda: [0, 0])
    for row in metrics:
        for target in ("5pct", "10pct"):
            apcer, bpcer = float(row[target + "_dev_apcer"]), float(row[target + "_dev_bpcer"])
            key = (target, row["patience"], row["network"], row["bn_state"], repr(apcer))
            points[key][0] += 1
            points[key][1] += int(apcer == 0.0 and bpcer == 1.0)
    write(out / "operating_points.csv",
          ["target", "patience", "network", "bn_state", "achieved_dev_apcer", "checkpoint_states", "all_attack"],
          [[*key, *points[key]] for key in sorted(points, key=lambda k: (k[0] != "5pct", int(k[1]), k[2], k[3], float(k[4])))])
    print(f"det_mean.csv: {len(order)} rows; operating_points.csv: {len(points)} rows; "
          f"all-attack operating points: {sum(v[1] for v in points.values())}")


if __name__ == "__main__":
    main()
