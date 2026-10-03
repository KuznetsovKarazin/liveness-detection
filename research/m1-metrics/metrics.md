# M1 — PAD metrics: definitions and conventions

Scope: the metric module of LivenessLab (`src/tesi_app/evaluation.py`, `BiometricMetrics`, `compute_metrics` and `metrics_by_attack_type`), the per-attack-type table (`scripts/eval_dataset.py --by-attack-type`), the checks that verify it (`scripts/check_metrics.py`) and the source-dev threshold protocol (`scripts/select_threshold.py`). The Conventions and the Threshold selection protocol were fixed before any M1 result was computed; the other sections were completed as each part was verified.

## Conventions

These conventions are fixed before writing code. Every check, table and report of M1 uses them.

### Labels, score direction, unit
- Label `y = 0` bona fide, `y = 1` attack.
- Score `s` = probability of attack, in [0, 1] (analyzers that return a bona fide probability are converted with `1 − p_bona_fide`).
- Unit of evaluation: one image (frame). For NUAA each image is a separate row; `n`, `n_bona_fide` and `n_attack` are always reported next to the metrics. For datasets with videos the aggregation rule is: mean of the attack scores of the frames of a video, then one decision per video. This rule is defined here but not applied in M1 (NUAA has no videos in the evaluated subset).

### Operational decision
- Attack if `s > threshold`; a score exactly equal to the threshold is bona fide (tie = bona fide).
- The inclusive rule `s >= threshold` is computed only in the diagnostic checks (a separate column or function), never in the tables.

### Constant and near-constant scores
- Scores are constant only if `max − min == 0` exactly (`np.ptp(s) == 0`); no tolerance. Near-constant scores (e.g. `[0.5, 0.5000001]`) are ordinary scores and every metric is computed.
- Constant scores with both classes present: AUC = 0.5 (the standard value for a ROC with a single operating point); EER = 0.5 with EER threshold not defined (`None`); BPCER@APCER10 not defined (`None`) because every threshold gives APCER 0 or 1. Fixed-threshold metrics (APCER, BPCER, ACER, accuracy) remain computable. Reports write the counts next to every undefined value.

### Missing class
- No attack samples: APCER `None`; no bona fide samples: BPCER `None`; in both cases ACER, AUC, EER and BPCER@APCER10 are `None`. `n_bona_fide` and `n_attack` are always reported. A metric that cannot be computed is never reported as 0.

### EER
- Computed on the full ROC curve (`drop_intermediate=False`): the curve point with the smallest `|BPCER − APCER|`, value `(BPCER + APCER) / 2`, no interpolation. On ties of `|BPCER − APCER|` the first point in the order returned by the ROC (decreasing threshold) is taken.
- The EER threshold is reported for information only and is never used as operational threshold in the tables.
  *Clarification added on 30 September 2026, before any development score was computed: this sentence refers to the EER threshold computed on the evaluated (test) set. The source-dev protocol below uses the EER threshold computed on the development set as operational threshold; it is applied with the operational rule `s > threshold`.*
- As a diagnostic comparison (not in the tables) this document also reports the EER obtained with linear interpolation between the two adjacent ROC points where `BPCER − APCER` changes sign.

### BPCER@APCER10
- Threshold `t` = the attack score at index `floor(n_attack · 0.10)` of the attack scores sorted in increasing order (empirical 0.10 quantile, no interpolation).
- BPCER@APCER10 = share of bona fide samples with `s >= t`.
- Not defined (`None`) when `n_attack · 0.10 < 1` (fewer than 10 attacks) or when a class is missing.
- Example: 10 attack scores `0.10, 0.20, …, 1.00` → index `floor(10 · 0.10) = 1` → `t = 0.20`; bona fide scores `0.05, 0.15, 0.25, 0.35` → two of four are `>= 0.20` → BPCER@APCER10 = 0.5.

### Per-attack-type metrics
- APCER is also reported per attack type with its denominator (number of attacks of that type); BPCER is global (bona fide samples have no attack type). A type with no samples has APCER `None` with denominator 0. NUAA has a single type, `print`.

### Source-dev threshold
- The threshold is chosen on the source development set only, with a criterion declared before any result is seen, frozen in `threshold.json` and applied unchanged to the test set. The test set is never used to choose it. See "Threshold selection protocol".

## Definitions and formulas

Notation: `B` = set of bona fide samples (`y = 0`), `A` = set of attack samples (`y = 1`), `n_bona_fide = |B|`, `n_attack = |A|`, `n = n_bona_fide + n_attack`, `s_i` = attack score of sample `i`, `t` = operational threshold (0.5 in every table unless stated otherwise). `#{…}` counts samples.

| Metric | Numerator | Denominator | Not defined (`None`) when |
|---|---|---|---|
| APCER | `#{i ∈ A : s_i <= t}` (attacks accepted as bona fide) | `n_attack` | `n_attack = 0` |
| BPCER | `#{i ∈ B : s_i > t}` (bona fide rejected as attacks) | `n_bona_fide` | `n_bona_fide = 0` |
| ACER | `APCER + BPCER` | 2 | APCER or BPCER is `None` |
| Accuracy | `#{i ∈ A : s_i > t} + #{i ∈ B : s_i <= t}` (correct decisions) | `n` | `n = 0` |
| AUC | `#{(a, b) ∈ A × B : s_a > s_b} + ½ · #{(a, b) ∈ A × B : s_a = s_b}` | `n_attack · n_bona_fide` | a class is missing |
| EER | see below | — | a class is missing |
| BPCER@APCER10 | `#{i ∈ B : s_i >= t10}` | `n_bona_fide` | a class is missing, `n_attack · 0.10 < 1`, or constant scores |
| APCER of type `k` | `#{i ∈ A_k : s_i <= t}` | `n_attack(k)` (attacks of type `k`) | `n_attack(k) = 0` |

Notes on the formulas:

- **AUC** is the Mann–Whitney statistic: the probability that a randomly drawn attack has a higher attack score than a randomly drawn bona fide sample, with ties counted as one half. It equals the area under the ROC curve built with the inclusive rule below. With constant scores and both classes every pair is a tie, so AUC = 0.5.
- **ROC curve and EER.** The ROC is built from the full list of distinct scores taken as candidate thresholds `θ` in decreasing order, preceded by `θ = +∞` (no sample rejected). At each `θ` the curve uses the inclusive rule `s >= θ` → attack, as scikit-learn's `roc_curve` does:
  - `BPCER(θ) = #{i ∈ B : s_i >= θ} / n_bona_fide` (false positive rate);
  - `APCER(θ) = #{i ∈ A : s_i < θ} / n_attack` (false negative rate, `1 − TPR`).
  
  The EER point is the curve point with the smallest `|BPCER(θ) − APCER(θ)|` (first point in decreasing-threshold order on ties); `EER = (BPCER(θ*) + APCER(θ*)) / 2`; the EER threshold is `θ*`, or `None` if `θ* = +∞`. The inclusive rule belongs to the threshold-free metrics (ROC, EER, BPCER@APCER10) only: the tables use the operational rule `s > t` at the fixed threshold `t`, so the EER threshold and the operational threshold are different objects (see the stepped-ROC example below).
- **Interpolated EER (diagnostic only).** Let `P1 = (BPCER_1, APCER_1)` and `P2 = (BPCER_2, APCER_2)` be the two adjacent ROC points where `d = BPCER − APCER` changes sign (`d_1 < 0 < d_2` or the reverse). With `λ = −d_1 / (d_2 − d_1)`, the interpolated EER is `BPCER_1 + λ · (BPCER_2 − BPCER_1)`, which equals `APCER_1 + λ · (APCER_2 − APCER_1)`. If some point has `d = 0`, both definitions give the same value. It is never reported in the tables.
- **BPCER@APCER10.** Sort the attack scores in increasing order, `a_(0) <= a_(1) <= … <= a_(n_attack − 1)`; `idx = floor(n_attack · 0.10)`; `t10 = a_(idx)`. The attacks with `s < t10` are accepted, so the realised APCER at `t10` is at most `idx / n_attack <= 0.10` (strictly smaller when `a_(idx)` is tied with lower scores). Bona fide samples with `s >= t10` are rejected.
- **Invalid input.** `compute_metrics` and `metrics_by_attack_type` raise `ValueError` for labels other than 0 and 1 (a value such as 0.7 is never truncated to 0), for NaN or infinite scores and for inputs of different lengths; callers drop the rows without a score before computing metrics.
- **Per-type APCER.** `A_k` = attacks of type `k`. The global APCER is the pooled value `Σ_k accepted_k / Σ_k n_attack(k)`, i.e. the average of the per-type APCER weighted by `n_attack(k)`, not their plain mean. BPCER is global. A metric that is `None` is written `—` in the tables, always next to its counts.

### Thresholds derived from scores and the operational rule

*Items (a) and (b) below are the previous text of this section, kept for traceability. They were **superseded on 2 October 2026 by the Amendment** at the end of "Threshold selection protocol": the selector no longer takes the ROC point or the quantile, and `threshold.json` no longer has the fields of the inclusive rule. The current text follows them.*

> *(superseded)* A threshold derived from scores (the EER threshold or the 10 % quantile threshold of the source dev set) is itself a score value, and some samples can have exactly that score. The ROC point uses the inclusive rule (`s >= θ` → attack), while the operational rule applied to the test set is `s > threshold` (tie = bona fide). Therefore:
>
> - **(a) EER threshold.** A sample whose score equals the EER threshold counts as an attack at the ROC point and as bona fide under the operational rule. `threshold.json` reports the declared value, the dev counts and errors under the operational rule, the number of dev samples tied at that value, and the values under the inclusive rule `>=` for comparison. Example (the data of worked example 5): EER threshold `θ* = 0.5`, one attack tied at 0.5. Inclusive rule (ROC point): APCER = 1/4 = 0.2500, BPCER = 1/3 = 0.3333. Operational rule `s > 0.5`: the tied attack is accepted → APCER = 2/4 = 0.5000, BPCER = 1/3 = 0.3333. Ties at the value: 1 (attack), 0 (bona fide).
> - **(b) 10 % quantile threshold.** With `t10 = a_(idx)`, `idx = floor(0.10 · n_attack)`, the operational rule accepts the attacks with `s <= t10`, i.e. `idx + 1` attacks when `a_(idx)` has no ties. The realised APCER on the dev set is therefore `(floor(0.10 · n_attack) + 1) / n_attack`, slightly above 10 % (with ties at `t10`, higher still). Example: 394 dev attacks → `idx = floor(39.4) = 39` → 40/394 = 0.1015 (10.15 %). The realised value is written in `threshold.json` next to the target.

**Current text (from 2 October 2026).** A threshold derived from scores is itself a score value, and some samples can have exactly that score. The ROC of the metric module uses the inclusive rule (`s >= θ` → attack), while the operational rule is `s > threshold` (tie = bona fide). The two objects are therefore different:

- **Metrics on the ROC (EER, EER threshold, BPCER@APCER10).** They keep the inclusive rule and the definitions above; at the EER threshold of the ROC a tied sample counts as an attack, while under the operational rule it counts as bona fide (worked example 5: the same number 0.5 gives APCER 1/4 at the ROC point and 1/2 with `s > 0.5`). These metrics are reported in the tables as threshold-free figures and are never used as operational thresholds.
- **The source-dev selector (`select_threshold.py select`).** Since the amendment of 2 October 2026 (see "Threshold selection protocol"), the selector does not take a ROC point: it evaluates every candidate threshold with the operational rule `s > t` and picks one of them. The dev counts recorded in `threshold.json` are exactly the counts of the selected point, and they do not change when the value is applied: no tied sample can move from one side to the other between selection and application. `select` re-reads `threshold.json` from disk and recomputes the counts to check it.

Worked example (computed by hand). Bona fide score 0.1, attack score 0.9 (`n_bona_fide = 1`, `n_attack = 1`). At a candidate `t`, `fn(t)` = attacks with `s <= t` (accepted), `fp(t)` = bona fide with `s > t` (rejected).

| candidate `t` | fn | fp | APCER | BPCER | `\|fp · n_attack − fn · n_bona_fide\|` |
|---|---|---|---|---|---|
| 0.9 | 1 | 0 | 1 | 0 | 1 |
| 0.1 | 0 | 0 | 0 | 0 | 0 |
| `nextafter(0.1, −∞)` = 0.09999999999999999 | 0 | 1 | 0 | 1 | 1 |

The criterion `eer` picks `t = 0.1`: APCER 0, BPCER 0, operational dev EER (APCER + BPCER)/2 = 0 (`dev_balanced_error_at_value`). The ROC of the module has its EER at `θ* = 0.9` (EER 0, ROC rule `s >= 0.9`); before the amendment the selector took that value, and with `s > 0.9` the attack at 0.9 was accepted (APCER 1). The ROC EER is still recorded in `threshold.json`, labelled as information only.

## Worked examples

All values below were computed by hand with exact fractions; decimals are rounded to four places as in the tables.

### 1. Tie at the threshold
Labels `y = [0, 0, 1, 1]`, scores `s = [0.5, 0.2, 0.5, 0.9]`, `t = 0.5`.
- Operational rule `s > 0.5`: bona fide 0.5 and 0.2 are accepted → BPCER = 0/2 = 0; attack 0.5 is accepted, 0.9 rejected → APCER = 1/2 = 0.5000; ACER = (1/2 + 0)/2 = 1/4 = 0.2500; accuracy = 3/4 = 0.7500.
- Diagnostic rule `s >= 0.5` (never in the tables): BPCER = 1/2, APCER = 0/2 = 0, ACER = 1/4, accuracy = 3/4.
The two rules give different APCER and BPCER; only the samples with `s = t` change side.

### 2. Constant scores with both classes
`y = [0, 0, 1, 1]`, `s = [0.3, 0.3, 0.3, 0.3]`, `t = 0.5`.
- All samples are accepted: APCER = 2/2 = 1, BPCER = 0/2 = 0, ACER = 1/2, accuracy = 2/4 = 1/2.
- AUC: the 2 · 2 = 4 pairs are all ties → (0 + ½ · 4)/4 = 1/2 = 0.5000.
- ROC points: `θ = +∞` → (BPCER 0, APCER 1); `θ = 0.3` → (1, 0). Both have `|BPCER − APCER| = 1`; the first is taken → EER = (0 + 1)/2 = 0.5000, EER threshold `None` (`θ* = +∞`).
- BPCER@APCER10: `None` (constant scores; here also `n_attack · 0.10 = 0.2 < 1`). Counts reported: `n_bona_fide = 2`, `n_attack = 2`.
- This is not a missing class (example 4): with both classes AUC and EER are defined (0.5); only the EER threshold and BPCER@APCER10 are `None`.

### 3. Near-constant scores
`y = [0, 1]`, `s = [0.5, 0.5000001]`, `t = 0.5`.
- `max − min = 1e-7 ≠ 0`: the scores are not constant and every metric is computed.
- Operational rule: bona fide 0.5 accepted (tie), attack 0.5000001 rejected → APCER = 0/1 = 0, BPCER = 0/1 = 0, ACER = 0, accuracy = 2/2 = 1.
- AUC: one pair, `s_a > s_b` → 1/1 = 1.0000.
- ROC points: `+∞` → (0, 1); `0.5000001` → (0, 0); `0.5` → (1, 0). Minimum `|BPCER − APCER| = 0` at `θ = 0.5000001` → EER = 0.0000, EER threshold 0.5000001.
- BPCER@APCER10: `None` (`n_attack · 0.10 = 0.1 < 1`).
The reference code's tolerance-based test (`np.allclose`) calls these scores constant and loses AUC and EER; the module uses the exact test and reports the values above (see "Differences from the reference code", point 4).

### 4. Missing class
`y = [1, 1, 1]`, `s = [0.2, 0.7, 0.9]`, `t = 0.5`.
- APCER = 1/3 = 0.3333 (the attack at 0.2 is accepted); BPCER `None` (`n_bona_fide = 0`); ACER `None`; accuracy = 2/3 = 0.6667; AUC, EER and BPCER@APCER10 `None`. Counts reported: `n_bona_fide = 0`, `n_attack = 3`. BPCER is never written as 0.

### 5. EER on a stepped ROC, with the interpolated diagnostic value
Bona fide scores `[0.1, 0.3, 0.6]`, attack scores `[0.2, 0.5, 0.7, 0.8]` (`n_bona_fide = 3`, `n_attack = 4`).

| θ | BPCER(θ) = #{b >= θ}/3 | APCER(θ) = #{a < θ}/4 | \|BPCER − APCER\| |
|---|---|---|---|
| +∞ | 0 | 1 | 1 |
| 0.8 | 0 | 3/4 | 3/4 |
| 0.7 | 0 | 1/2 | 1/2 |
| 0.6 | 1/3 | 1/2 | 1/6 |
| 0.5 | 1/3 | 1/4 | 1/12 |
| 0.3 | 2/3 | 1/4 | 5/12 |
| 0.2 | 2/3 | 0 | 2/3 |
| 0.1 | 1 | 0 | 1 |

- No point has BPCER = APCER. The minimum is at `θ* = 0.5`: EER = (1/3 + 1/4)/2 = 7/24 = 0.2917 (reported value); EER threshold 0.5. At `θ*` with the inclusive rule: APCER 1/4, BPCER 1/3.
- Interpolated (diagnostic): `d` changes sign between `θ = 0.6` (`d_1 = 1/3 − 1/2 = −1/6`) and `θ = 0.5` (`d_2 = 1/3 − 1/4 = 1/12`); `λ = (1/6) / (1/12 + 1/6) = 2/3`; EER = 1/3 + 2/3 · (1/3 − 1/3) = 1/3 = 0.3333 (check: 1/2 + 2/3 · (1/4 − 1/2) = 1/3). The two values differ by 1/24 on this small curve; only 0.2917 is reported.
- Operational threshold `t = 0.5` with `s > t` on the same data: attacks 0.2 and 0.5 accepted → APCER = 2/4 = 0.5000; bona fide 0.6 rejected → BPCER = 1/3 = 0.3333; ACER = 5/12 = 0.4167; accuracy = 4/7 = 0.5714. The same number 0.5 used as ROC threshold (`s >= θ`) and as operational threshold (`s > t`) gives different APCER (1/4 vs 1/2) because the attack at exactly 0.5 changes side.
- AUC = (1 + 2 + 3 + 3)/12 = 9/12 = 0.7500 (for each attack, the number of bona fide with a lower score).

### 6. Quantile index for BPCER@APCER10
- 12 attacks, sorted: `0.05, 0.10, 0.10, 0.30, 0.45, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95`; `idx = floor(12 · 0.10) = floor(1.2) = 1` → `t10 = 0.10`. Attacks with `s < 0.10`: one → realised APCER = 1/12 = 0.0833 (below 0.10 because of the tie at 0.10). Bona fide `0.02, 0.08, 0.10, 0.40, 0.95`: three are `>= 0.10` → BPCER@APCER10 = 3/5 = 0.6000.
- 9 attacks: `9 · 0.10 = 0.9 < 1` → `None`.
- The 10-attack example in "Conventions" gives `idx = 1`, `t10 = 0.20`, realised APCER 1/10, BPCER@APCER10 = 2/4 = 0.5.

### 7. Per-type APCER
Bona fide scores `[0.5, 0.6, 0.1]`; attacks: `print` `[0.5, 0.9]`, `replay` `[0.2, 0.7]`, `mask` `[0.8]`; expected types also include `3d`; `t = 0.5`.
- BPCER = 1/3 (0.6 rejected; 0.5 is a tie, accepted).
- `3d`: `n_attack` 0, APCER `None`; `mask`: 0/1 = 0; `print`: 1/2 (0.5 accepted); `replay`: 1/2 (0.2 accepted).
- Global APCER = (0 + 1 + 1)/(1 + 2 + 2) = 2/5 = 0.4000, not the plain mean of the per-type values ((0 + 1/2 + 1/2)/3 = 1/3).

## Per-attack-type reporting

- Function: `metrics_by_attack_type(y_true, y_scores, attack_types, threshold=0.5, expected_types=None)` in `src/tesi_app/evaluation.py`. It applies the operational rule `s > threshold` (tie = bona fide) and returns `threshold`, `n_bona_fide`, `n_attack`, the global `bpcer` with `rejected_bona_fide`, and `by_type` = `{type: {n_attack, accepted, apcer}}` in alphabetical order. The type is read only for attack rows; a type listed in `expected_types` with no samples appears with `n_attack` 0 and `apcer` `None`. Mismatched lengths, labels other than 0/1 (including non-integer values such as 0.7), non-finite scores, or an attack row without a type (missing, NaN, empty or whitespace only) raise `ValueError`.
- Table: `scripts/eval_dataset.py --by-attack-type --manifest <manifest.csv>` appends a second Markdown table (analyzer × attack type: `n_attack`, `accepted`, `apcer`, plus `n_bona_fide`, `rejected_bona_fide`, `bpcer`) after the main table. The main table is unchanged; without the option the output is byte-identical to the previous version. The manifest must have the columns `file` (the cache key, e.g. `attack/<name>.jpg`) and `label`, and may have `attack_type`. The scored samples of every analyzer and the manifest rows must be exactly the same set, with the same labels: a row more or less, or a different label, stops the script (exit code 1). Attack types must belong to a fixed vocabulary; variants such as `Print` are refused. The group's `score-schema.json` declares `attack_type` as a string without a list of values, so the vocabulary is LivenessLab's own list, `print`, `cut_photo`, `replay`, `mask`, `unknown` (the types of the evaluated datasets, plus `unknown` for an attack without type); it is not a list of the group. If a later version of the schema declares a list, that list is used, plus `unknown`. A bona fide row with an attack type is an error.
- **NUAA:** a single type, `print`: all NUAA attacks are printed photographs. The C1 manifest has no `attack_type` column, so the type `print` is assigned to every attack row, as in the `scores.csv` of C1. For each of the 30 scored analyzers the table reports `print` with denominator 150 and an APCER equal to the global APCER of the main table; BPCER has denominator 150.
- Other datasets: types come from the manifest's `attack_type` column; if the column is missing or empty for an attack row, the type is `unknown` and the table note gives how many attacks have no type.

## Unit of evaluation and aggregation

- Unit: one image. On NUAA (C1 evaluation set) `n = 300`: 150 bona fide and 150 attack images, each a separate row; every table reports `n`, `n_bona_fide` and `n_attack`, and per-type tables report the denominator of each type.
- Images of the same subject and session are not statistically independent (they are frames of the same capture); the counts are counts of images, not of subjects or captures, and no metric is presented as if the rows were independent draws.
- Videos (not applied in M1): the attack score of a video is the mean of the attack scores of its frames; one decision per video with the same operational rule; the video is then the unit and the counts are counts of videos.
- No other aggregation (across analyzers, datasets or seeds) is performed in M1.

## Differences from the reference code

The formulas of `BiometricMetrics` derive from `src/evaluation_utils.py` of the reference repository (MIT licence; `src/livedetection/` in the development repository), at commit `4334db59d87855f73a060e2f6bc220e707003a72`. The line numbers below refer to that file at that commit. The differences were verified by reading that code and are covered by the cases of `scripts/check_metrics.py`; no claim is made about results published with it. The differences are:

1. **BPCER@APCER direction.** The reference counts bona fide samples with score `<=` the threshold, i.e. the accepted ones (`calculate_bpcer_at_apcer`, line 129: `bonafide_scores <= threshold`), and so returns `1 − BPCER` (a perfect classifier gets 100 %). Here the rejected ones are counted (`>=`).
2. **Missing class.** When a class is missing the reference returns 0.0 for APCER or BPCER (lines 54 and 66) and for BPCER@APCER (line 120), and NaN for AUC (lines 248 and 257, `ModelEvaluator`) and EER (lines 87–88). The module returns `None` for APCER or BPCER, ACER, AUC, EER and BPCER@APCER10, and always reports `n_bona_fide` and `n_attack` (example 4). A metric that cannot be computed is never 0.
3. **Full ROC for the EER.** The reference searches the EER on `roc_curve` with `drop_intermediate=True` (line 90: `roc_curve(y_true, y_scores)`, scikit-learn's default), which drops collinear points; the point where APCER and BPCER meet can be among the dropped ones, and the EER becomes the mean of two different errors. Here the full curve (`drop_intermediate=False`) is used. The smallest `|BPCER − APCER|` is found by comparing exact integer counts, `|fp · n_attack − fn · n_bona_fide|` (fp = rejected bona fide, fn = accepted attacks at the curve point), so that an exact tie between two curve points is not broken by floating-point rounding; the first point in decreasing-threshold order wins the tie, as stated in the Conventions.
4. **Exact constancy.** The reference tests constancy with `np.allclose(s, s[0])` (line 45, `_is_constant_scores`), which treats near-constant scores such as `[0.5, 0.5000001]` as constant and returns NaN for AUC and EER. The module uses the exact test `max − min == 0`: near-constant scores are ordinary scores and every metric is computed (example 3: AUC 1, EER 0).
5. **Constant scores with both classes.** The reference returns NaN for AUC and EER (same guards as in point 2) and 0.0 for BPCER@APCER (line 112). The module returns AUC = 0.5 and EER = 0.5 (the single-point ROC), with EER threshold `None` and BPCER@APCER10 `None`; the fixed-threshold metrics are computed as usual (example 2).

## Threshold selection protocol

Declared on 30 September 2026, before any development score is computed.

1. **Development set (NUAA).** A subject-wise partition of the official NUAA training split (`bonafide_training`, `attack_training`; subject = the 4-digit prefix of the file name). The dev subjects are exactly the validation subjects held out when the NUAA "official" CNNs were trained (`GroupShuffleSplit`, 20 % of the subjects, seed 42): those images were used only for early stopping, never to fit the weights. The dev set contains all training-split images of those subjects. The list of files, with the SHA-256 of each image, is saved in `nuaa_dev_manifest.csv`, and its hash is stored in `threshold.json`. The official test split (`*_validation`), from which the 300 evaluation images come, is never read by the selection.
2. **Separation checks** (automatic, exit code 1 on failure): no dev image in the C1 manifest (by SHA-256) or in the official test split (by SHA-256); no dev subject among the subjects used to fit the weights. NUAA is not subject-disjoint between its official train and test splits (the nine training subjects appear in both, in different sessions): the separation with the test set is by official split and session, and this limit is stated in every report.
3. **Analyzers.** Only analyzers whose source (training) dataset is NUAA with this subject-wise validation receive a NUAA source-dev threshold. For every other analyzer the NUAA training split is not its source dev set; those rows are reported at the fixed threshold 0.5 only, with a note.
4. **Criterion (primary).** Threshold at the EER of the dev scores (EER threshold as defined above). **Alternative (documented, not used unless chosen before the run):** the threshold that gives APCER = 10 % on the dev set (the BPCER@APCER10 threshold defined above).
5. **Freeze and apply.** `threshold.json` records criterion, dev manifest hash, analyzer, value, commit and date. The value is applied unchanged to the 300 test images. The test table is reported twice, clearly labelled: at the historical threshold 0.5 (continuity with C1) and at the source-dev threshold.
6. **Authorisation.** Scoring the dev images is new inference on images and runs only with explicit authorisation. Until then the code is verified on synthetic scores and on a fictitious cache, and `threshold.json` is not produced from real data.

### Clarifications (30 September 2026, after the code review, before the authorised dev scoring)

These clarifications do not change the criterion, the definition of the development set or the analyzers above; they state facts about the data that the protocol relies on.

- **(a) Use of the dev images in training.** Of the 574 dev images, 493 were among the images sampled for training and were used only for early stopping (validation loss); the other 81 were never read by the training. None was used to fit the weights.
- **(b) Separation from the test set.** It is by official split, acquisition session and image hash. "Session" here is the acquisition session in the fifth field of the NUAA file name (`01` or `02` in the training split, `03` in the test split). It is not the `session` column of the C1 manifest nor the `session_id` column of `scores.csv`, which hold the third field of the name and are kept unchanged from C1.
- **(c) Shared identities.** The two dev subjects also appear among the 300 C1 test images, as attacks of session `03` (22 of the 150 attacks). No image is shared (checked by SHA-256); the identity is shared.
- **(d) Size of the dev set.** The dev set has only two subjects (574 images, 180 bona fide and 394 attacks). The threshold chosen on it will carry high uncertainty.
- **(e) Wording.** In point 2 of the protocol, "subjects 1–9" was replaced by "the nine training subjects" (same subjects; wording only).
- **(f) Test split.** The official test split is never used to choose the threshold. `check-separation` hashes its images and `select` lists its file names (and applies the rule on the fifth field of the name) only to refuse any test image.
- **(g) Non-informative dev scores.** If the dev AUC of an analyzer is below 0.5 or its dev EER above 0.5 (either condition is enough), no score inversion is applied: the threshold is still applied, and the row of the `apply` table is flagged `non-informative` with the reason (`dev AUC < 0.5`, `dev EER > 0.5`, or both). `apply` also reports, as a sensitivity analysis, the test metrics without the 22 attacks of the two dev subjects; the table names no subject identifier.

### Amendment (2 October 2026, after the review of PR #2, before any authorised dev score)

The review of PR #2 found that the selector chose the threshold on the inclusive ROC (`s >= θ`) or at the 0.10 quantile of the attack scores, while the threshold is applied with `s > threshold`, so that the selected point could be lost (example in "Thresholds derived from scores and the operational rule"). The protocol above and its clarifications are unchanged except for the selection rule and the EER used by clarification (g) (replaced by the balanced error at the selected threshold, see "Non-informative flag" below), both stated here; no development score had been computed when this amendment was written. The primary criterion is the EER, understood as the balance of APCER and BPCER on the dev set only under the operational rule, as agreed with the reviewer.

- **Operational rule (unchanged).** Attack iff `s > t` (tie = bona fide). At a threshold `t`: accepted attacks `fn(t) = #{i ∈ A : s_i <= t}`, rejected bona fide `fp(t) = #{i ∈ B : s_i > t}`; APCER = `fn / n_attack`, BPCER = `fp / n_bona_fide`. Selection and application use the same rule.
- **Candidates.** Every distinct dev score of the analyzer (bona fide and attacks), plus one value strictly below the minimum score, `t_low = nextafter(min_score, −∞)` (the next smaller double), which accepts every sample. `t_low` is included only if `t_low >= 0.0`: scores live in [0, 1] and the values of `threshold.json` must stay in [0, 1]. If the minimum score is exactly 0.0 the candidate below it is not representable and is omitted. Between two consecutive candidates the counts do not change, so these candidates cover every operating point reachable with `s > t` on the dev scores.
- **Criterion `eer` (primary).** The candidate minimising `|fp · n_attack − fn · n_bona_fide|`, computed on integers (exact; it equals `n_attack · n_bona_fide · |BPCER − APCER|`). Tie-break: the highest candidate threshold. `threshold.json` records the value, the counts, APCER and BPCER at it under the operational rule, and `dev_balanced_error_at_value` = (APCER + BPCER)/2 at the selected value, which under this criterion is the operational dev EER. The EER of the module's ROC (`compute_metrics`, inclusive rule) is recorded as `roc_eer_information_only`: it is not the basis of the selection.
- **Criterion `apcer10` (alternative).** Among the candidates with `10 · fn <= n_attack` (integer test: APCER <= 0.10 under the operational rule), the one with the smallest `fp` (BPCER); tie-break: the highest threshold. Because `fn` grows and `fp` decreases with `t`, this is the largest feasible candidate. It is defined whenever both classes are present; with fewer than 10 attacks it means that no attack is accepted. If no candidate is feasible (possible only when attack scores equal 0.0, where `t_low` is omitted) `select` stops with an error. This criterion is a constrained rule on the operational counts; it is **not** the quantile threshold of the table metric BPCER@APCER10, which keeps its definition above (inclusive rule, `t10 = a_(floor(0.10 · n_attack))`) and is reported unchanged as `bpcer_at_apcer10_metric`.
- **Degenerate cases.** An analyzer whose dev scores are missing or contain a single class stops `select` with an error. Constant dev scores (value `c > 0`) have the candidates `c` and `t_low`: under `eer` both give `|·| = n_attack · n_bona_fide`, the tie goes to `c` (every sample accepted: APCER 1, BPCER 0, operational dev EER 0.5); under `apcer10` only `t_low` is feasible (APCER 0, BPCER 1; dev ACER at the threshold 0.5). Inverted scores can give an exact balance with both errors high (bona fide 0.8, 0.9, attacks 0.1, 0.2: `t = 0.2`, APCER 1, BPCER 1); the row is then flagged as below.
- **Non-informative flag.** Clarification (g) flags a row when the dev AUC is below 0.5 or when the balanced error at the selected threshold, `dev_balanced_error_at_value` = (APCER + BPCER)/2 on the dev set with `s > value`, is above 0.5. Under criterion `eer` this value is the operational dev EER and is labelled "operational dev EER"; under criterion `apcer10` the threshold does not balance the two errors, so the same value is not an EER and is labelled "dev ACER at the threshold" (column caption and reason of the flag). Above 0.5 the threshold does worse, averaged over the two classes, than the rule that accepts every sample (APCER 1, BPCER 0, balanced error 0.5). The ROC EER is not used for the flag.
- **Round trip.** After writing `threshold.json`, `select` re-reads it from disk, recomputes the counts with `s > value` for every analyzer on the dev scores and stops with an error if any count differs from the selected one. `apply` recomputes the whole selection in memory from the dev cache and refuses a `threshold.json` whose entries differ.
- **Entry of `threshold.json`.** `criterion`, `value`, `decision_rule`, `selection_rule` (the rule above in one line), `dev` = {`at_value_operational_rule_gt` (counts, APCER, BPCER, ACER, ties at the value), `dev_balanced_error_at_value`, `roc_eer_information_only`, `auc`, `bpcer_at_apcer10_metric`}, `fingerprint`, `note`. The former fields of the inclusive rule (`at_value_roc_point_rule_ge`, `eer_threshold`) are removed.
- **Hand-derived cases** (exact counts; they are the selector cases of `scripts/check_select_threshold.py`, with `fn`, `fp` as defined above):
  1. Bona fide 0.1, attack 0.9 → `eer`: `t = 0.1`, APCER 0/1, BPCER 0/1 (table in "Thresholds derived from scores and the operational rule").
  2. Bona fide 0.5, attack 0.5000001 → `eer`: candidates 0.5000001 (fn 1, fp 0, objective 1), 0.5 (0, 0, 0), `nextafter(0.5, −∞)` (0, 1, 1) → `t = 0.5`, APCER 0/1, BPCER 0/1.
  3. Attacks 0.1, 0.2, …, 1.0 (`n_attack = 10`), bona fide 0.05, 0.15, 0.25, 0.35 → `apcer10`: feasible means `fn <= 1`; `fn(0.15) = 1` (only 0.1), `fn(0.2) = 2`, so the feasible candidates are `nextafter(0.05, −∞)`, 0.05, 0.10, 0.15, with `fp` = 4, 3, 3, 2 → `t = 0.15`, APCER 1/10, BPCER 2/4 = 0.5000. Before the amendment the quantile gave `t = 0.20`, with `s > 0.20` two attacks accepted (APCER 2/10).
  4. Same data, `eer`: objective `|10 · fp − 4 · fn|`; 0.35 (fn 3, fp 0) → 12; 0.30 (3, 1) → 2; 0.25 (2, 1) → 2; 0.20 (2, 2) → 12; the others are larger. Tie between 0.30 and 0.25 → the higher, `t = 0.30`: APCER 3/10, BPCER 1/4, operational dev EER 11/40 = 0.2750.
  5. Bona fide 0.2, 0.5, 0.5, attacks 0.5, 0.5, 0.8 (ties across classes at 0.5, all bona fide under `s > 0.5`) → `eer`: 0.8 → 9, 0.5 (fn 2, fp 0) → 6, 0.2 (fn 0, fp 2) → 6, `t_low` → 9; tie → `t = 0.5`, APCER 2/3, BPCER 0. `apcer10` (`fn = 0`): `t = 0.2`, BPCER 2/3.
  6. Bona fide 0.2, 0.6, attacks 0.4, 0.8, 0.9 (`n_attack = 3 < 10`) → `apcer10`: `fn = 0`, `t = 0.2`, APCER 0, BPCER 1/2.
  7. Constant scores 0.3 (2 + 2) → `eer`: `t = 0.3` (APCER 1, BPCER 0); `apcer10`: `t = 0.29999999999999993` (APCER 0, BPCER 1). Constant scores 0.0: `eer` gives `t = 0.0` (APCER 1), `apcer10` has no feasible candidate.
  8. Bona fide 0.3, 0.7, 0.8, attacks 0.2, 0.9 (`n_attack = 2`, `n_bona_fide = 3`) → `eer`, objective `|2 · fp − 3 · fn|`: 0.9 (fn 2, fp 0) → 6; 0.8 (1, 0) → 3; 0.7 (1, 1) → 1; 0.3 (1, 2) → 1; 0.2 (1, 3) → 3; `t_low` (0, 3) → 6. Exact tie 0.7 / 0.3 → `t = 0.7`, APCER 1/2, BPCER 1/3. The objective must be compared on integers: in floating point `|fp/3 − fn/2|` is 1/6 at both points but rounds differently and would pick 0.3.
