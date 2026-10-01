# M1 — Metric check cases and hand derivations

Checks: `scripts/check_metrics.py` (standard-library `unittest`, run with `python scripts/check_metrics.py [--report PATH] [--require-restricted]`; exit code 0 when every case passes, 1 otherwise; with `--require-restricted` a missing restricted input is a failure, not a skip). Conventions: `metrics.md`, section "Conventions". The expected values of the synthetic cases are derived by hand as exact fractions; the real-data cases (`real_nuaa`, `round_trip_nuaa_minifasnet`) use the independent reference implementation of the check script, which is itself checked against the hand-derived fractions in case `reference_self_check`. No expected value comes from scikit-learn or from the module under test.

## Notation and rules used in every derivation

- `B` = bona fide scores (`y = 0`), `A` = attack scores (`y = 1`), `n_b = |B|`, `n_a = |A|`.
- Operational decision at threshold `t` (default 0.5): attack if `s > t`. APCER = `#{a in A : a <= t} / n_a` (attacks accepted), BPCER = `#{b in B : b > t} / n_b` (bona fide rejected), ACER = (APCER + BPCER) / 2, accuracy = correct decisions / n.
- ROC (full curve, as `roc_curve(drop_intermediate=False)`): thresholds `+inf` followed by every distinct score in decreasing order; at ROC threshold `u` a sample is called attack if `s >= u`. At each point BPCER(u) = `#{b >= u} / n_b`, APCER(u) = `#{a < u} / n_a`, `d(u) = BPCER(u) − APCER(u)`.
- EER: the ROC point with the smallest `|d|`; on ties the first point in decreasing-threshold order; value `(BPCER + APCER) / 2`; EER threshold = that point's `u` (`None` when `u = +inf`).
- EER, interpolated (diagnostic only, never in tables): linear interpolation between the two adjacent ROC points where `d` changes sign from negative to positive: `λ = −d0 / (d1 − d0)`, EER = `BPCER0 + λ (BPCER1 − BPCER0)` (equal to the interpolated APCER). If a point has `d = 0` it is that point's value.
- AUC = Mann–Whitney statistic: over the `n_a · n_b` pairs (attack, bona fide), count 1 when attack > bona fide and 1/2 on ties, divide by `n_a · n_b`.
- BPCER@APCER10: undefined when `n_a · 0.10 < 1` (i.e. `n_a < 10`), when a class is missing or when the scores are constant; otherwise `t10` = sorted(A)[`floor(n_a · 0.10)`] and BPCER@APCER10 = `#{b >= t10} / n_b`.
- Constant scores: `max − min == 0` exactly.
- Comparison in the checks: counts as integers; fractions against the module's float with absolute tolerance 1e-12 (float division is correctly rounded, so the noise is around 1e-16, while the smallest meaningful difference is the AUC step with half-ties, `1 / (2 · n_a · n_b)`: 1/45,000 ≈ 2.2e-5 on NUAA (150 attacks and 150 bona fide), larger in the synthetic cases); EER thresholds by exact float equality (they are input scores); `None` must be exactly `None`.
- The module names the bona fide count `n_real` (key kept: it is used by the app and by the table generator); the checks accept `n_real` or `n_bona_fide` (name used by the conventions and by the table column).
- Non-finite scores (NaN, ±inf) are invalid input: `compute_metrics` and `metrics_by_attack_type` raise `ValueError` (cases `invalid_compute_metrics` and `invalid_metrics_by_attack_type`). Callers drop them upstream (the cache filter of `summarize()` keeps only finite scores, and the C1 consistency checker rejects them), so no metric is ever computed on them.

## Case `perfect` — perfect separation

Input: `y = [0, 0, 0, 1, 1, 1]`, `s = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]`, `t = 0.5`. `B = {0.1, 0.2, 0.3}`, `A = {0.7, 0.8, 0.9}`.

- APCER = attacks `<= 0.5`: none → **0/3**. BPCER = bona fide `> 0.5`: none → **0/3**. ACER **0**. Accuracy **6/6 = 1**.
- ROC points (u: BPCER, APCER, d): +inf: 0, 3/3, −1 · 0.9: 0, 2/3, −2/3 · 0.8: 0, 1/3, −1/3 · **0.7: 0, 0, 0** · 0.3: 1/3, 0 · 0.2: 2/3, 0 · 0.1: 1, 0.
- EER: minimum `|d| = 0` at u = 0.7 → **EER 0, threshold 0.7**. Interpolated EER: 0 (point with d = 0).
- AUC: every one of the 3 · 3 = 9 pairs has attack > bona fide → **9/9 = 1**.
- BPCER@APCER10: `n_a = 3 < 10` → **None**.

## Case `inverted` — every decision wrong

Input: `y = [0, 0, 0, 1, 1, 1]`, `s = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]`. `B = {0.7, 0.8, 0.9}`, `A = {0.1, 0.2, 0.3}`.

- APCER = attacks `<= 0.5`: 3 → **3/3 = 1**. BPCER = bona fide `> 0.5`: 3 → **3/3 = 1**. ACER **1**. Accuracy **0/6 = 0**.
- ROC: +inf: 0, 1, −1 · 0.9: 1/3, 1, −2/3 · 0.8: 2/3, 1, −1/3 · **0.7: 1, 1, 0** · 0.3: 1, 2/3, +1/3 · 0.2: 1, 1/3 · 0.1: 1, 0.
- EER: `|d| = 0` at u = 0.7 → **EER (1 + 1)/2 = 1, threshold 0.7**. Interpolated EER: 1.
- AUC: no pair with attack > bona fide, no ties → **0/9 = 0**.
- BPCER@APCER10: `n_a = 3` → **None**.

## Case `constant_two_classes` — constant scores, both classes

Input: `y = [0, 0, 1, 1]`, `s = [0.7, 0.7, 0.7, 0.7]`. `max − min = 0` → constant.

- Fixed threshold (still computable): APCER = attacks `<= 0.5`: 0 → **0/2**; BPCER = bona fide `> 0.5`: 2 → **2/2 = 1**; ACER **1/2**; accuracy **2/4 = 1/2**.
- ROC: +inf: 0, 1, d = −1 · 0.7: 1, 0, d = +1. Both points have `|d| = 1`; the first (+inf) is taken → **EER (0 + 1)/2 = 1/2, threshold None**. This is exactly the convention (EER 0.5, threshold undefined). Interpolated EER: λ = 1/2 → 1/2.
- AUC: 2 · 2 = 4 pairs, all ties → 4 · 1/2 / 4 = **1/2** (convention: 0.5 for a single-point ROC).
- BPCER@APCER10: constant scores → **None** (also `n_a = 2 < 10`).

## Case `constant_ten_attacks` — constant scores with enough attacks for the quantile

Input: `y = [0]·2 + [1]·10`, `s = [0.7]·12`. Here `n_a = 10` would make the quantile defined, so only the constant-score rule makes BPCER@APCER10 undefined (the plain quantile rule would give t10 = 0.7 and BPCER 2/2 = 1).

- APCER **0/10**, BPCER **2/2 = 1**, ACER **1/2**, accuracy **10/12 = 5/6**.
- ROC: +inf: 0, 1 · 0.7: 1, 0 → tie on `|d| = 1`, first point → **EER 1/2, threshold None**. Interpolated 1/2.
- AUC: 20 pairs, all ties → **10/20 = 1/2**.
- BPCER@APCER10: constant → **None**.

## Case `constant_only_bona_fide` and `constant_only_attacks` — constant scores with one class

- `y = [0, 0, 0]`, `s = [0.7, 0.7, 0.7]`: APCER **None**; BPCER = bona fide `> 0.5`: 3 → **3/3 = 1**; ACER **None**; accuracy **0/3 = 0**; EER, EER threshold, AUC, BPCER@APCER10 **None** (missing class takes precedence over the constant rule: nothing becomes 0.5).
- `y = [1, 1]`, `s = [0.3, 0.3]`: APCER = attacks `<= 0.5`: 2 → **2/2 = 1**; BPCER **None**; ACER **None**; accuracy **0/2 = 0**; EER, EER threshold, AUC, BPCER@APCER10 **None**.

## Case `constant_at_threshold` — constant scores equal to the threshold, both classes

Input: `y = [0, 0, 1, 1]`, `s = [0.5, 0.5, 0.5, 0.5]`, `t = 0.5`. Every score ties the threshold → every sample is called bona fide.
- APCER = attacks `<= 0.5`: 2 → **2/2 = 1**; BPCER = bona fide `> 0.5`: 0 → **0/2**; ACER **1/2**; accuracy **2/4 = 1/2**.
- ROC: +inf: 0, 1 · 0.5: 1, 0 → tie on `|d| = 1`, first point → **EER 1/2, threshold None**; interpolated 1/2. AUC: 4 ties of 4 pairs → **1/2**. BPCER@APCER10 **None**.

## Case `near_constant` — `[0.5, 0.5000001]`, labels `[0, 1]`

Input: `y = [0, 1]`, `s = [0.5, 0.5000001]`. `max − min = 1e-7 ≠ 0` → ordinary scores.

- APCER: attack 0.5000001 > 0.5 → rejected → **0/1**. BPCER: bona fide 0.5 is not `> 0.5` → accepted → **0/1**. ACER **0**. Accuracy **2/2 = 1**.
- ROC: +inf: 0, 1, −1 · **0.5000001: 0, 0, 0** · 0.5: 1, 0, +1 → **EER 0, threshold 0.5000001**. Interpolated 0.
- AUC: 1 pair, attack > bona fide → **1/1 = 1**.
- BPCER@APCER10: `n_a = 1` → **None**.

## Case `ties_at_threshold` — scores exactly equal to 0.5 in both classes

Input: `y = [0, 0, 0, 0, 1, 1, 1, 1]`, `s = [0.2, 0.5, 0.5, 0.8, 0.3, 0.5, 0.9, 0.95]`. `B = {0.2, 0.5, 0.5, 0.8}`, `A = {0.3, 0.5, 0.9, 0.95}`.

Operational rule `s > 0.5` (through `compute_metrics`):
- APCER = attacks `<= 0.5`: {0.3, 0.5} → **2/4 = 1/2**. BPCER = bona fide `> 0.5`: {0.8} → **1/4**. ACER = (1/2 + 1/4)/2 = **3/8**. Accuracy = (3 bona fide accepted + 2 attacks rejected)/8 = **5/8**.

Diagnostic rule `s >= 0.5` (case `decisions_at`, through `BiometricMetrics.decisions_at(y_scores, 0.5, inclusive=True)`):
- Decisions: `[0, 1, 1, 1, 0, 1, 1, 1]`. APCER = attacks `< 0.5`: {0.3} → **1/4**. BPCER = bona fide `>= 0.5`: {0.5, 0.5, 0.8} → **3/4**.
- Strict rule through the same function (`inclusive=False`, and the default): decisions `[0, 0, 0, 1, 0, 0, 1, 1]`, APCER **2/4**, BPCER **1/4** (same as `compute_metrics`).

Threshold-free metrics:
- ROC: +inf: 0, 1, −1 · 0.95: 0, 3/4, −3/4 · 0.9: 0, 1/2, −1/2 · **0.8: 1/4, 1/2, −1/4** · 0.5: 3/4, 1/4, +1/2 · 0.3: 3/4, 0, +3/4 · 0.2: 1, 0, +1.
- EER: min `|d| = 1/4` at u = 0.8 → **EER (1/4 + 1/2)/2 = 3/8, threshold 0.8**.
- Interpolated EER: sign change between u = 0.8 (d = −1/4) and u = 0.5 (d = +1/2): λ = (1/4)/(3/4) = 1/3; BPCER = 1/4 + 1/3 · (3/4 − 1/4) = 5/12; APCER = 1/2 − 1/3 · (1/2 − 1/4) = 5/12 → **5/12 ≈ 0.4167** (diagnostic, vs 3/8 = 0.375 without interpolation).
- AUC (16 pairs): attack 0.3 beats {0.2} → 1; attack 0.5 beats {0.2} and ties the two 0.5 → 1 + 2 · 1/2 = 2; 0.9 → 4; 0.95 → 4. Total 11 → **11/16**.
- BPCER@APCER10: `n_a = 4` → **None**.

## Case `only_bona_fide`

Input: `y = [0, 0, 0]`, `s = [0.2, 0.6, 0.4]`.
- APCER **None** (no attacks). BPCER = {0.6} → **1/3**. ACER **None**. Accuracy = 2/3 (accuracy stays computable with one class present: it is not listed among the undefined metrics of the conventions). EER, EER threshold, AUC, BPCER@APCER10 **None**. `n_attack = 0`.

## Case `only_attacks`

Input: `y = [1, 1, 1, 1]`, `s = [0.2, 0.6, 0.7, 0.9]`.
- APCER = {0.2} → **1/4**. BPCER **None**. ACER **None**. Accuracy **3/4**. EER, EER threshold, AUC, BPCER@APCER10 **None**. `n_bona_fide = 0`.

## Case `empty`

Input: `y = []`, `s = []`. `n = n_bona_fide = n_attack = 0`; every metric, accuracy included, **None**.

## Case `bpcer10_five_attacks`

Input: `B = [0.05, 0.15, 0.25, 0.35]`, `A = [0.1, 0.2, 0.3, 0.4, 0.5]`. `n_a · 0.10 = 0.5 < 1` → **None**.

## Case `bpcer10_ten_attacks` — worked example of `metrics.md`

Input: `B = [0.05, 0.15, 0.25, 0.35]`, `A = [0.1, 0.2, …, 1.0]` (10 values). Index `floor(10 · 0.10) = 1` → t10 = sorted(A)[1] = 0.2. Bona fide `>= 0.2`: {0.25, 0.35} → **2/4 = 1/2**.

## Case `bpcer10_twenty_attacks_tie` — bona fide score equal to t10

Input: `B = [0.1, 0.15, 0.3, 0.9]`, `A = [0.05, 0.10, …, 1.00]` (20 values, step 0.05, written as literals so that 0.15 is the same float in both lists). Index `floor(20 · 0.10) = 2` → t10 = sorted(A)[2] = 0.15. Bona fide `>= 0.15`: {0.15, 0.3, 0.9} → **3/4** (with `>` it would be 2/4: this case fixes the `>=` of the convention).

## Case `bpcer10_nine_attacks` — boundary of the definition

Input: `B = [0.05, 0.15, 0.25, 0.35]`, `A = [0.1, 0.2, …, 0.9]` (9 values). `n_a · 0.10 = 0.9 < 1` → **None** (10 attacks is the smallest defined case, see `bpcer10_ten_attacks`).

## Case `bpcer10_fifteen_attacks_floor` — floor, not ceil or round

Input: `B = [0.02, 0.12, 0.5, 0.9]`, `A = [0.05, 0.10, …, 0.75]` (15 values, step 0.05). `n_a · 0.10 = 1.5`: the convention takes `floor(1.5) = 1` → t10 = sorted(A)[1] = 0.10; bona fide `>= 0.10`: {0.12, 0.5, 0.9} → **3/4**. With `ceil(1.5) = 2` or `round(1.5) = 2` the threshold would be sorted(A)[2] = 0.15 and the value {0.5, 0.9} → 2/4: the bona fide score 0.12, between the two candidate thresholds, separates the rules.

## Case `stepped_roc` — ties inside both classes

Input: `B = [0.1, 0.4, 0.4, 0.6, 0.6]`, `A = [0.4, 0.6, 0.8, 0.8, 0.8]`, `t = 0.5`.
- Fixed threshold: APCER = attacks `<= 0.5`: {0.4} → **1/5**; BPCER = bona fide `> 0.5`: {0.6, 0.6} → **2/5**; ACER **3/10**; accuracy = (3 + 4)/10 = **7/10**.
- ROC (steps of 2/5 because of the ties): +inf: 0, 1, −1 · 0.8: 0, 2/5, −2/5 · **0.6: 2/5, 1/5, +1/5** · 0.4: 4/5, 0, +4/5 · 0.1: 1, 0, +1.
- EER: min `|d| = 1/5` at u = 0.6 → **EER (2/5 + 1/5)/2 = 3/10, threshold 0.6**.
- Errors at the EER point (`apcer_at_eer`, `bpcer_at_eer`: inclusive rule `s >= 0.6`, as the ROC): attacks `< 0.6`: {0.4} → **APCER 1/5**; bona fide `>= 0.6`: {0.6, 0.6} → **BPCER 2/5** (their mean is the EER, 3/10). With the strict rule `s > 0.6` the same threshold would give APCER 2/5 (the attack at 0.6 is accepted) and BPCER 0: the check distinguishes the two rules.
- Interpolated EER (diagnostic): sign change between u = 0.8 (d = −2/5) and u = 0.6 (d = +1/5): λ = (2/5)/(3/5) = 2/3; BPCER = 0 + 2/3 · 2/5 = 4/15; APCER = 2/5 − 2/3 · 1/5 = 4/15 → **4/15 ≈ 0.2667** (vs 3/10 = 0.3 on the curve point).
- AUC (25 pairs): attack 0.4 beats {0.1}, ties two 0.4 → 1 + 1 = 2; attack 0.6 beats {0.1, 0.4, 0.4}, ties two 0.6 → 3 + 1 = 4; each attack 0.8 beats all 5 → 15. Total 21 → **21/25**.
- BPCER@APCER10: `n_a = 5` → **None**.

## Case `eer_tie_break` — two points with the same `|BPCER − APCER|`

Input: `y = [0, 0, 1, 1, 1]`, `s = [0.4, 0.7, 0.3, 0.6, 0.9]`. `B = {0.4, 0.7}`, `A = {0.3, 0.6, 0.9}`.
- Fixed threshold: APCER = {0.3} → **1/3**; BPCER = {0.7} → **1/2**; ACER **5/12**; accuracy = (1 + 2)/5 = **3/5**.
- ROC: +inf: 0, 1, −1 · 0.9: 0, 2/3, −2/3 · **0.7: 1/2, 2/3, −1/6** · 0.6: 1/2, 1/3, +1/6 · 0.4: 1, 1/3, +2/3 · 0.3: 1, 0, +1.
- u = 0.7 and u = 0.6 tie exactly on `|d| = 1/6`; the convention takes the first in decreasing-threshold order → **EER (1/2 + 2/3)/2 = 7/12, threshold 0.7**. (The other point would give 5/12.) In floating point `|0.5 − (1 − 1/3)| = 0.16666666666666674` and `|0.5 − (1 − 2/3)| = 0.16666666666666663`, so a plain float `argmin` picks the second point: the tie must be resolved on exact counts (`|fp · n_a − fn · n_b|`).
- Interpolated EER: λ = (1/6)/(1/3) = 1/2 → **1/2**.
- AUC (6 pairs): attack 0.3 → 0; 0.6 beats {0.4} → 1; 0.9 → 2. Total 3 → **3/6 = 1/2**.
- BPCER@APCER10: `n_a = 3` → **None**.

## Case `eer_tie_float_key` — an exact EER tie that a float key breaks the wrong way

Input: `y = [0, 0, 0, 1, 1]`, `s = [0.15, 0.0, 0.45, 0.2, 0.1]`. `B = {0.0, 0.15, 0.45}` (`n_b = 3`), `A = {0.1, 0.2}` (`n_a = 2`), `t = 0.5`.
- Fixed threshold: every score is `<= 0.5`, so every sample is accepted: APCER = 2/2 = **1**; BPCER = 0/3 = **0**; ACER **1/2**; accuracy = 3/5 (the three bona fide) → **3/5**.
- ROC (u: BPCER, APCER, d): +inf: 0, 1, −1 · 0.45: 1/3, 1, −2/3 · **0.2: 1/3, 1/2, −1/6** · 0.15: 2/3, 1/2, +1/6 · 0.1: 2/3, 0, +2/3 · 0.0: 1, 0, +1.
- u = 0.2 and u = 0.15 tie exactly on `|d| = 1/6` (integer key `|fp · n_a − fn · n_b|`: `|1 · 2 − 1 · 3| = 1` and `|2 · 2 − 1 · 3| = 1`); the first in decreasing-threshold order wins → **EER (1/3 + 1/2)/2 = 5/12, threshold 0.2**. The other point would give (2/3 + 1/2)/2 = 7/12. In floating point `|1/3 − 1/2| = 0.16666666666666669` and `|2/3 − 1/2| = 0.16666666666666663`, so an `argmin` on the float `|fpr − fnr|` picks u = 0.15 and reports 7/12: this case fails if the key is not computed on exact counts.
- Errors at the EER point (inclusive rule `s >= 0.2`): attacks `< 0.2`: {0.1} → **APCER 1/2**; bona fide `>= 0.2`: {0.45} → **BPCER 1/3**.
- Interpolated EER: sign change between u = 0.2 (d = −1/6) and u = 0.15 (d = +1/6): λ = (1/6)/(1/3) = 1/2; BPCER = 1/3 + 1/2 · 1/3 = **1/2** (APCER = 1/2 + 1/2 · 0 = 1/2).
- AUC (6 pairs): attack 0.2 beats {0.0, 0.15} → 2; attack 0.1 beats {0.0} → 1; no ties. Total 3 → **3/6 = 1/2**.
- BPCER@APCER10: `n_a = 2` → **None**.

## Case `default_threshold` — `compute_metrics` called without the threshold

Input: `y = [0, 0, 1, 1]`, `s = [0.2, 0.495, 0.5, 0.8]`, no `threshold` argument: the default must be `t = 0.5` and the output must report `threshold` 0.5.
- BPCER: bona fide `> 0.5`: none → **0/2 = 0**; APCER: attacks `<= 0.5`: {0.5} (tie = bona fide) → **1/2**; ACER **1/4**; accuracy = (2 + 1)/4 = **3/4**.
- A different default changes these values without the real cache: with 0.49 the bona fide at 0.495 is rejected (BPCER 1/2) and the attack at 0.5 is detected (APCER 0).
- The same data without `threshold` in `metrics_by_attack_type` (case `attack_types_default_threshold`, types `["", "", "print", "print"]`): **threshold 0.5**, **BPCER 0/2 = 0**, **`print` APCER 1/2**.

## Cases `invalid_compute_metrics` and `invalid_metrics_by_attack_type` — inputs that must raise `ValueError`

No metric is computed on invalid input: each call below must raise `ValueError` (any other outcome, including a different exception, fails the check).
- `compute_metrics`: a label 2; a label −1; a label 0.7 (NumPy array and plain list: it would otherwise be truncated to 0); text labels `"0"`/`"1"`; a NaN score; a `+inf` score; a `−inf` score; a NaN score with attacks only and an infinite score with bona fide only (one class: neither AUC nor ROC is computed, so the error must come from the input check of the module); lengths 3 and 4.
- `metrics_by_attack_type` (four samples, types `["", "", "print", "print"]` unless stated): a label 2; a label −1; a label 0.7; a NaN score; an attack type `float("nan")`; an attack type `np.float64("nan")`; an attack type `None`; an attack type `""`; an attack type `"   "` (whitespace only); an `attack_types` list of length 2.

## Case `decisions_at`

Interface: a `staticmethod` of `BiometricMetrics`, `decisions_at(y_scores, threshold, inclusive=False)`, returning an array of 0/1 decisions (1 = attack): `s > threshold` when `inclusive=False` (operational rule, the default), `s >= threshold` when `inclusive=True` (diagnostic only).

Diagnostic rule `>=` through `BiometricMetrics.decisions_at(y_scores, threshold, inclusive=...)` on the data of `ties_at_threshold` (derivation above): inclusive → decisions `[0, 1, 1, 1, 0, 1, 1, 1]`, APCER 1/4, BPCER 3/4; strict and default → `[0, 0, 0, 1, 0, 0, 1, 1]`, APCER 2/4, BPCER 1/4. APCER and BPCER are counted in the check directly from the returned decisions.

## Case `compact_roc` — content of `out["roc"]`

`compute_metrics` returns a compact ROC for the plot: points `[BPCER, 1 − APCER]` of `roc_curve` (default `drop_intermediate=True`), one every `max(1, len // 60)` points, rounded to 4 decimals, followed by `[1.0, 1.0]`. On the data of `eer_tie_break` the curve has 6 points (step 1) and no three consecutive points are aligned, so no point is dropped and the list is deterministic. From the ROC of that case, with `1 − APCER = TP/3`: +inf → [0, 0]; 0.9 → [0, 1/3]; 0.7 → [1/2, 1/3]; 0.6 → [1/2, 2/3]; 0.4 → [1, 2/3]; 0.3 → [1, 1]; plus the final [1, 1]. Rounded: **`[[0.0, 0.0], [0.0, 0.3333], [0.5, 0.3333], [0.5, 0.6667], [1.0, 0.6667], [1.0, 1.0], [1.0, 1.0]]`**, compared by exact equality.

## Case `attack_types` — `metrics_by_attack_type`

Input: bona fide `[0.1, 0.5, 0.6, 0.2, 0.3]` (attack type `""`, ignored); `print` attacks `[0.4, 0.5, 0.7, 0.9]`; `replay` attacks `[0.51, 0.8, 0.3]`; `threshold = 0.5`, `expected_types = ["print", "replay", "mask"]`.
- `threshold` 0.5; `n_bona_fide` 5; `n_attack` 7.
- Bona fide `> 0.5`: {0.6} → `rejected_bona_fide` **1**, `bpcer` **1/5**.
- `print`: 4 attacks, accepted (`<= 0.5`) {0.4, 0.5} → `accepted` 2, `apcer` **2/4**.
- `replay`: 3 attacks, accepted {0.3} → `accepted` 1, `apcer` **1/3** (0.51 > 0.5 is rejected).
- `mask`: expected type without samples → `n_attack` **0**, `accepted` 0, `apcer` **None**.
- `by_type` keys exactly {`mask`, `print`, `replay`}.

## Cases `round_trip_synthetic` and `round_trip_nuaa_minifasnet`

Scores are written with `repr(float(s))` to a CSV (`label,score_attack`) in a `tempfile.TemporaryDirectory` (removed at the end), read back with `csv.DictReader` and `float()`. Labels and scores are first normalised to Python lists of `int` and `float` (so NumPy arrays are accepted). Checks: the re-read lists are identical to the originals, and every key of `compute_metrics` (including `roc`) is exactly equal (`==`, no tolerance) between the in-memory and the re-read scores. Synthetic scores: `0.1 + 0.2`, `1/3`, `2/3`, `0.5000001`, `0.5` (twice), `0.49999999999999994`, `1e-17`, `0.9999999999999999`, `0.7`, `0.12345678901234568`, `0.8` with labels `[0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0]`. The second case uses the 300 `minifasnet` scores of `$LIVENESSLAB_RESULTS_DIR/eval/nuaa.json` (skipped if the cache is absent; failed instead with `--require-restricted`).

## Case `real_nuaa` — rows of the NUAA cache

Source (read-only): `$LIVENESSLAB_RESULTS_DIR/eval/nuaa.json` (default `results/eval/nuaa.json`), `scores[analyzer][file] = {"y", "s", ...}`, rows with a finite score (same filter as `summarize()`), `t = 0.5`. For each analyzer that has a numeric row in `$LIVENESSLAB_RESULTS_DIR/c1/nuaa_table.md` (30 rows, `minifasnet` first as the case asked for) the expected values are computed by the reference implementation in the check script: direct counts for APCER/BPCER/accuracy, the ROC over the sorted distinct scores with exact tie-breaking on integer counts for the EER and for the errors at its point (`apcer_at_eer`, `bpcer_at_eer`, inclusive rule), Mann–Whitney pair counting (via sorted bona fide scores and `bisect`) for the AUC, the `floor(n_a/10)` index for BPCER@APCER10. They are compared (1) with `compute_metrics` at 1e-12 and (2) with the C1 table formatted as `f"{v:.4f}"` (the same format as the table; no value of these sample sizes lies on a 4-decimal rounding tie: denominators 150 and 300 for the rates, 45 000 for the AUC because of the half-ties, i.e. `2 · 150 · 150`; `k / 45 000 · 10^4 = 2k/9` is never an odd multiple of 1/2). With the table present, case `real_nuaa:table_rows` first requires its numeric rows to be exactly the analyzers of the cache with finite scores: 30 rows, no duplicate, no row absent from the cache, no analyzer of the cache without a row, note rows only for analyzers without scores, no unrecognised row. If the cache is absent (exported package) the case is skipped with a message, and the summary ends with `RESULT: PASS (N cases skipped: restricted inputs not found)`. If the cache is present but `c1/nuaa_table.md` is absent, only the table comparison is skipped (one `SKIP` line per analyzer) and the analyzers checked are all those of the cache with at least two finite scores. With `--require-restricted` every such skip becomes a failure (exit code 1); the export of the package uses this option.

`minifasnet` (n = 300, 150 + 150): APCER 0/150, BPCER 0/150, ACER 0, accuracy 300/300, EER 0 at threshold 0.988560848752968, AUC 22500/22500 = 1, BPCER@APCER10 0/150.

The reference implementation itself is checked in case `reference_self_check`: it must reproduce exactly every hand-written fraction of the synthetic cases above, including the interpolated EER values.

## Interpolated EER (diagnostic, not in tables)

| case | EER (curve point) | EER (linear interpolation) |
|---|---|---|
| perfect | 0 | 0 |
| inverted | 1 | 1 |
| constant_two_classes | 1/2 | 1/2 |
| constant_ten_attacks | 1/2 | 1/2 |
| constant_at_threshold | 1/2 | 1/2 |
| near_constant | 0 | 0 |
| ties_at_threshold | 3/8 = 0.375 | 5/12 ≈ 0.4167 |
| stepped_roc | 3/10 = 0.3 | 4/15 ≈ 0.2667 |
| eer_tie_break | 7/12 ≈ 0.5833 | 1/2 |
| eer_tie_float_key | 5/12 ≈ 0.4167 | 1/2 |

## Run on the module before the fixes (30 September 2026)

`.venv/bin/python scripts/check_metrics.py` on `src/tesi_app/evaluation.py` as found (with `metrics_by_attack_type` already present): 26 test cases, 867 checks, 12 failing checks in 6 cases, exit code 1.

| case | metric | expected | obtained | cause |
|---|---|---|---|---|
| constant_two_classes | eer, auc | 1/2, 1/2 | None, None | constant scores are treated as non-computable |
| constant_ten_attacks | eer, auc | 1/2, 1/2 | None, None | same |
| constant_at_threshold | eer, auc | 1/2, 1/2 | None, None | same |
| near_constant | eer, eer_threshold, auc | 0, 0.5000001, 1 | None, None, None | `np.allclose` treats `[0.5, 0.5000001]` as constant |
| eer_tie_break | eer, eer_threshold | 7/12, 0.7 | 5/12, 0.6 | float `argmin` on `|FPR − FNR|` breaks an exact tie towards the second point |
| decisions_at | decisions_at | function | not available | the diagnostic `>=` function does not exist yet |

All other cases pass, including the 30 real NUAA rows against the independent implementation and against the C1 table at 4 decimals.
