# NUAA table recomputed with the verified metric module: comparison with C1

## Purpose

M1 changed the metric module (`src/tesi_app/evaluation.py`: exact constancy test, AUC and EER defined for constant scores with both classes, EER point chosen on integer counts with ties resolved to the first point in decreasing threshold order, `decisions_at` as the single definition of the decision rule; see `metrics.md`). This document checks that the C1 NUAA table, recomputed from the same cached scores with the new module, is unchanged, and explains every difference found below the four printed decimals.

Inputs (not modified by M1):

- score cache `results/eval/nuaa.json` (30 analyzers with a score x 300 images, 150 bona fide and 150 attacks);
- C1 table `results/c1/nuaa_table.md`, exported CSV `results/c1/nuaa_scores.csv`, manifest `results/c1/nuaa_manifest.csv`, configuration `results/c1/nuaa_config.json`, group-schema scores `results/c1/20260928-C1-seed42-a80f6f9/scores.csv`. The C1 table was produced at commit `a80f6f9` (see its header).

No new training; no development-set score was computed for the delivered materials or appears in them: dev scoring is pending authorisation. Every number below comes from the cached scores.

## Commands

The commands of this section were run from the root of the operator's development repository (private), where `results/` is the default `LIVENESSLAB_RESULTS_DIR` and `.venv/` the environment. Commit `a80f6f9` cited below is the development commit at which the C1 table was produced; in the group repository C1 is the merge `7731076`, and `git show a80f6f9:…` works only in the development repository. From this package, the same checks are run with the commands of `README.md` (paths relative to `livenesslab/`, results under `livenesslab/results/`).

Recompute the table and the exported CSV from the cache:

```bash
.venv/bin/python scripts/eval_dataset.py --dataset nuaa --from-cache --lang en --export results/m1/nuaa_scores.csv --table results/m1/nuaa_table.md
```

APCER by attack type, written to a separate file (the plain table above is the one read by the consistency check):

```bash
.venv/bin/python scripts/eval_dataset.py --dataset nuaa --from-cache --lang en --by-attack-type --manifest results/c1/nuaa_manifest.csv --table results/m1/nuaa_table_by_attack_type.md
```

Row-by-row comparison (header excluded: the first 8 lines contain title, date, command, commit and environment) and byte comparison of the CSV:

```bash
diff <(tail -n +9 results/c1/nuaa_table.md) <(tail -n +9 results/m1/nuaa_table.md) && echo "table rows identical"
cmp results/c1/nuaa_scores.csv results/m1/nuaa_scores.csv && echo "CSV identical"
```

Consistency check with self-test on the new output. The check reads a C1-like folder, so it runs on a temporary copy of `results/c1` in which only the table and the exported CSV are replaced (the group-schema `scores.csv` stays the C1 one, because the scores are the same); the report is written in the run folder of the copy and then saved as `results/m1/consistency_check_m1.txt`:

```bash
TMP_C1=$(mktemp -d)/c1
cp -R results/c1 "$TMP_C1"
cp results/m1/nuaa_table.md results/m1/nuaa_scores.csv "$TMP_C1"/
.venv/bin/python scripts/check_c1_consistency.py --dataset nuaa --c1 "$TMP_C1" --run-dir "$TMP_C1"/20260928-C1-seed42-a80f6f9 --self-test; echo "exit code $?"
cp "$TMP_C1"/20260928-C1-seed42-a80f6f9/consistency_check.txt results/m1/consistency_check_m1.txt
```

Full-precision comparison between the C1 module (commit `a80f6f9`) and the current one on the same score arrays (the C1 module is loaded from git as a temporary module; nothing in the repository is changed):

```bash
OLD=$(mktemp -d)
git show a80f6f9:src/tesi_app/evaluation.py > "$OLD/evaluation_c1.py"
.venv/bin/python - "$OLD/evaluation_c1.py" <<'EOF'
import importlib.util, json, sys
import numpy as np
sys.path.insert(0, "src")
from tesi_app import analyzers, evaluation as new  # noqa: F401
spec = importlib.util.spec_from_file_location("tesi_app.evaluation_c1", sys.argv[1])
old = importlib.util.module_from_spec(spec); old.__package__ = "tesi_app"; spec.loader.exec_module(old)
cache = json.load(open("results/eval/nuaa.json"))
keys = ["n", "n_real", "n_attack", "apcer", "bpcer", "acer", "eer", "eer_threshold", "bpcer_at_apcer10", "auc", "accuracy"]
for aid, m in new.summarize("nuaa", None, verify_fingerprints=False)["analyzers"].items():
    if "note" in m:
        continue
    rows = [(v["y"], v["s"]) for v in cache["scores"][aid].values() if v.get("s") is not None and np.isfinite(v["s"])]
    y = np.array([r[0] for r in rows]); s = np.array([r[1] for r in rows], dtype=float)
    a, b = old.compute_metrics(y, s), new.compute_metrics(y, s)
    d = {k: (repr(a[k]), repr(b[k])) for k in keys if a[k] != b[k]}
    if d:
        print(aid, d)
EOF
```

## Result at four decimals

Expected: no difference. Observed: **no difference**.

- All 35 table rows (30 rows with a score and 5 note rows: the four NUAA `-pooled` models excluded from their own dataset and `facemesh`, descriptive) are identical to `results/c1/nuaa_table.md`: `diff` of the table bodies is empty.
- `results/m1/nuaa_scores.csv` is byte-identical to `results/c1/nuaa_scores.csv` (9,000 rows, same SHA-256).
- `results/m1/nuaa_table_by_attack_type.md` repeats the same main table (identical rows) and adds one `print` row per analyzer with denominator 150; its APCER equals the global APCER on every row.

Only the header differs, by construction: date, command (output paths under `results/m1/` instead of `results/c1/`) and commit (the commit of this repository at which the table is regenerated, instead of `a80f6f9`). The tables are regenerated from a clean working tree (no "(uncommitted changes)" flag) by `scripts/export_m1.py` (development repository; not included in this folder), which records the commit in `run-report.md`.

## Why nothing changes

The M1 corrections act only in situations that do not occur in the NUAA cache, or that the C1 code already handled in the same way:

1. **No constant or near-constant row.** The new constancy test (`max − min == 0` exactly) and the new values for constant scores (AUC 0.5, EER 0.5) apply only when all scores are equal. On the 30 rows with a score, the smallest range `max − min` is 0.0673 (`livenessnet__casia_fasd`); the next ones are 0.1611 (`attacknet_v2_2__casia_fasd`) and 0.3118 (`attacknet_v1__celeba_spoof`). No row is close to the old `np.allclose` tolerance either, so no AUC or EER was set to "—" by C1 and none is set differently now.
2. **Exact ties at the threshold 0.5.** Only two analyzers have scores exactly equal to 0.5: `ida` 5 (4 bona fide, 1 attack) and `lbp` 2 (1 bona fide, 1 attack) (counted from the cache). C1 already applied the rule `score > 0.5` (tie = bona fide), which is the operational rule of M1 (`decisions_at(..., inclusive=False)`); APCER, BPCER, ACER and accuracy are therefore unchanged. The `>=` rule is computed only in the diagnostic checks, never in the table.
3. **Exact EER tie.** One row has two ROC points with the same `|BPCER − APCER|` on integer counts: `attacknet_v1__casia_fasd`, at the adjacent thresholds 0.5810228288173676 (24 bona fide rejected, 26 attacks accepted: EER (24 + 26)/300 = 1/6 = 0.1667) and 0.581022709608078 (24 and 22: 46/300 = 0.1533, four attacks tied at that score). The M1 convention takes the first point in decreasing threshold order; the C1 code (floating-point `argmin` on the full ROC) picked the same point. EER 0.1667 and its threshold are identical in both versions. With the other point the printed EER would have been 0.1533: this is the one row where the tie rule matters, and it is resolved identically.
4. **BPCER@APCER10 and AUC** are computed with the same formulas as in C1 (quantile index `floor(0.1 · 150) = 15`, bona fide rejected with score >= that value; `roc_auc_score` whenever the scores are not constant). They are identical at full precision on all 30 rows.

## Differences at full precision

Comparing the dictionaries returned by `compute_metrics` in the C1 module (`a80f6f9`) and in the current module on the same score arrays, **only the EER value changes, on 12 of the 30 rows**; APCER, BPCER, ACER, accuracy, AUC, BPCER@APCER10, the counts and the EER threshold are bit-for-bit identical on all 30 rows.

Cause: C1 took the EER as `(fpr + fnr)/2` with `fnr = 1 − tpr` from `roc_curve`, which adds rounding in the subtraction; the current module computes `(fp/n_bona_fide + fn/n_attack)/2` from integer counts. The chosen ROC point (and threshold) is the same on every row, so the exact value is the same; only the last bits of the floating-point result differ. The new value equals the correctly rounded exact fraction on all 12 rows.

| analyzer | bona fide rejected / 150 | attacks accepted / 150 | exact EER | C1 value | M1 value | difference |
|---|---|---|---|---|---|---|
| clip_zeroshot | 2 | 2 | 1/75 | 0.013333333333333322 | 0.013333333333333334 | 7 ulp |
| lbp | 48 | 48 | 8/25 | 0.31999999999999995 | 0.32 | 1 ulp |
| fourier | 10 | 10 | 1/15 | 0.06666666666666665 | 0.06666666666666667 | 1 ulp |
| ida | 45 | 45 | 3/10 | 0.30000000000000004 | 0.3 | 1 ulp |
| livenessnet__nuaa | 99 | 99 | 33/50 | 0.6599999999999999 | 0.66 | 1 ulp |
| attacknet_v2_2__celeba_spoof | 38 | 38 | 19/75 | 0.2533333333333333 | 0.25333333333333335 | 1 ulp |
| livenessnet__casia_fasd-pooled | 115 | 115 | 23/30 | 0.7666666666666666 | 0.7666666666666667 | 1 ulp |
| attacknet_v1__celeba_spoof | 58 | 58 | 29/75 | 0.3866666666666667 | 0.38666666666666666 | 1 ulp |
| livenessnet__synthaspoof | 99 | 99 | 33/50 | 0.6599999999999999 | 0.66 | 1 ulp |
| attacknet_v2_1__synthaspoof | 100 | 100 | 2/3 | 0.6666666666666667 | 0.6666666666666666 | 1 ulp |
| dog | 50 | 50 | 1/3 | 0.33333333333333337 | 0.3333333333333333 | 1 ulp |
| iqa | 51 | 51 | 17/50 | 0.33999999999999997 | 0.34 | 1 ulp |

(ulp = unit in the last place of the double-precision value.) The largest difference in ulp is 7 (`clip_zeroshot`, 1.2e-17); the largest absolute difference is 1.1e-16 (1 ulp, e.g. `livenessnet__nuaa`).

After the change the EER equals the correctly rounded exact fraction on 28 of 30 rows (16 in C1). The two remaining rows are unchanged from C1 and differ from the exact fraction by one rounding step of the final division, identically in both versions: `attacknet_v1__casia_fasd` 0.16666666666666669 (exact 1/6, 24 and 26 errors) and `attacknet_v2_2__celeba_spoof-pooled` 0.3833333333333333 (exact 23/60, 2 and 113 errors).

None of these differences can change a printed value: with 150 bona fide and 150 attacks every rate is a fraction with denominator 150 or 300, and every AUC value is a multiple of 1/45,000 (half-ties count 1/2 of a pair out of 150 · 150, so the denominator is 2 · 22,500); none of these fractions lies exactly halfway between two four-decimal values (a halfway point is an odd multiple of 1/20,000, and `k / 45,000 · 20,000 = 4k/9` is never odd), so a change of a few ulp cannot move the rounding.

## Consistency check

`scripts/check_c1_consistency.py --self-test`, run as above on the copy with the new table and CSV (checker at commit `e3fed80`, SHA-256 `3f1ee00fdbd603a3f8c453789470a22ff8c8ff66e066de7c2e598d44fc072c59`): **exit code 0**, "result: OK, no inconsistency" (300 samples, 30 analyzers with a score, 5 rows without score, 9,000 pairs in the exported CSV and in `scores.csv`, metrics recomputed and compared at four decimals on 30 rows); self-test: 17 faulty copies, all rejected with their expected message ("all faults detected"). That first report was written to `results/m1/consistency_check_m1.txt`, which the export now overwrites (see below). Its first line names the run folder of the copy (`20260928-C1-seed42-a80f6f9`), because the group-schema `scores.csv` checked is the C1 one. The checker delivered with this package is a later revision (strict table parser, values compared as text, fingerprints, every column of the group schema, configuration, manifest and cache hashes against the values recorded for C1, analyzers and checkpoints of the configuration, manifest subjects against the file names, per-type table; 80 faulty copies in the self-test). On the same inputs it gives the same result. The export reruns it at every regeneration, on the run folder (`consistency_check.txt`) and on a temporary copy of the C1 folder with the recomputed table, CSV and per-type table: the `consistency_check_m1.txt` delivered in the restricted run folder is that regenerated report (80 faulty copies, all rejected with their expected message). The figures above (17 faulty copies, checker at `e3fed80`) are those of the first run, kept as its record.

## SHA-256

| file | SHA-256 |
|---|---|
| `results/m1/nuaa_scores.csv` | `35391a8e9e64628f3d4f34bcb36311071f1b0926ffd0339fec19a374e8e32b05` |
| `results/c1/nuaa_table.md` | `9615804274a24d0b54e9ab02fd4d397747748475fcb404792b166de63bb386d0` |
| `results/c1/nuaa_scores.csv` | `35391a8e9e64628f3d4f34bcb36311071f1b0926ffd0339fec19a374e8e32b05` |
| `results/c1/nuaa_manifest.csv` | `86095da1531b1f24b1233c9668ac1ab37714978e444b313456732c028ccfb40c` |
| `results/c1/nuaa_config.json` | `f0dfdb90a1f61b86fa21e1519a742f180c632bdf677a6c9c45b022526ca8fa89` |
| `results/c1/20260928-C1-seed42-a80f6f9/scores.csv` | `0a52d875eaeabe61173162dd645ec415349822e9855572b0c5c22bf3db594ed0` |
| `results/eval/nuaa.json` (score cache) | `9b2b63b467e5e413c1fd5051e7c1c478f482301dc17070d60950d2e1daa34e65` |
| `src/tesi_app/evaluation.py` (current module) | `5deb4426538322807878895d5423a336b7a928288393b95e071cb35b783fa94e` |

The regenerated tables and the consistency report carry the date and commit of the regeneration in their header, so their SHA-256 changes at every regeneration even when the rows do not: the hashes of the tables are listed in `run-report.md` (field "APCER / BPCER / ACER with denominators"), and those of every file of the run, the consistency report included, in `SHA256SUMS` of the restricted run folder, not here. The two `nuaa_scores.csv` hashes are equal. The SHA-256 of `evaluation.py` is that of the delivered file: after the comparison above only a comment of its module docstring changed (the reference to this package's `metrics.md`), so the code compared is the same; the export checks that this hash matches the file it delivers. The C1 files and the score cache had the same SHA-256 before and after this work.
