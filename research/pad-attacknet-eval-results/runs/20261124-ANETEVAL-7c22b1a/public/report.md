# Evaluation 20261124-ANETEVAL-7c22b1a of the C5 comparison on the NUAA test

Status: **PASS**. Primary analysis: patience 4, threshold 0.5; individual 95% intervals: no correction for multiplicity, no simultaneous coverage over the contrasts, variations or comparisons

## Contrasts (variation of the AttackNet minus that of LivenessNet, ACER)

| AttackNet | computed | mean | 95% interval | valid resamples | reading |
|---|---|---|---|---|---|
| AttackNetV1 | yes | 15.14 | [4.96, 24.05] | 926 | ACER variation of AttackNetV1 minus that of LivenessNet higher than zero |
| AttackNetV2_1 | yes | 13.95 | [7.56, 21.63] | 926 | ACER variation of AttackNetV2_1 minus that of LivenessNet higher than zero |
| AttackNetV2_2 | yes | 11.61 | [4.46, 18.95] | 926 | ACER variation of AttackNetV2_2 minus that of LivenessNet higher than zero |

## Variations of ACER (re-estimated minus ordinary BN)

| network | computed | mean | 95% interval |
|---|---|---|---|
| LivenessNet | yes | -7.07 | [-13.33, -0.41] |
| AttackNetV1 | yes | 8.07 | [3.47, 11.66] |
| AttackNetV2_1 | yes | 6.87 | [3.35, 11.85] |
| AttackNetV2_2 | yes | 4.53 | [1.00, 8.82] |
| ResNet-18 | yes | 2.91 | [-1.73, 6.63] |

C2 reproduced exactly before the test of the new models: True. Not evaluable models: 0.

Command: `python -I -B scripts/pad_anet_eval.py evaluate --config delivery/aneteval/pad_anet_eval.json --out-root results/aneteval --runs-root results/anetrun --attempt 1 --i-am-authorized`
