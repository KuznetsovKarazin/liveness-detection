# Appendix B. Models that decide every test or dev frame as one class, and not evaluable models

Descriptive count of `PROTOCOL.md` §6 (f) on the test: per network, patience, BN state and threshold (0.5,
or the EER threshold of the dev of the fold in the same BN state), the models that decide every test frame
attack or every test frame bona fide ("k of n": k such models of the n evaluated). Computed by the script
from `units.csv` (APCER 0 and BPCER 1: every frame attack; APCER 1 and BPCER 0: every frame bona fide; both
classes are present in every fold) and checked against `test_one_class` of `results.json`; in
`single_class.csv`. It enters no reading.

| network | p4, ordinary, 0.5 | p4, ordinary, EER | p4, re-estimated, 0.5 | p4, re-estimated, EER | p12, ordinary, 0.5 | p12, ordinary, EER | p12, re-estimated, 0.5 | p12, re-estimated, EER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LivenessNet | 2 of 12 (2 attack) | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 |
| AttackNetV1 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 |
| AttackNetV2_1 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 |
| AttackNetV2_2 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 |
| ResNet-18 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 | 0 of 12 |

2 models decide every test frame as the same class (LivenessNet fold 1 seed 42, patience 4, ordinary BN, threshold
0.5: every frame attack; LivenessNet fold 4 seed 44, patience 4, ordinary BN, threshold 0.5: every frame attack; ACER
1/2); they enter the means like the others; both are inside the ordinary-BN mean of LivenessNet, and so inside its
variation and every contrast: nothing was excluded after the test was seen.

Not evaluable models: 0 (`not_evaluable` of `results.json`); every network has 12 of 12 fold-seed pairs evaluable in
both BN states at patience 4 and at patience 12 (guardian of `results.json`).

## On the dev (pre-registered, `PROTOCOL.md` §6 (f), first part)

Per network, patience and BN state, at threshold 0.5 on the dev of the fold of each model: "k of n evaluable
(a attack, b bona fide)", the models that decide every dev frame as one class. The dev count is computed by
the C2 rule in each run and read from `public/one_class.csv` of the eight run records (six of C5, two of C2),
checked against their `status.json`, the pins and `public/models.csv`; in `single_class.csv` (split `dev`).

| network | p4, ordinary | p4, re-estimated | p12, ordinary | p12, re-estimated |
|---|---:|---:|---:|---:|
| LivenessNet | 2 of 12 evaluable (2 attack, 0 bona fide) | 0 of 12 evaluable (0 attack, 0 bona fide) | 0 of 12 evaluable (0 attack, 0 bona fide) | 0 of 12 evaluable (0 attack, 0 bona fide) |
| AttackNetV1 | 0 of 12 evaluable (0 attack, 0 bona fide) | 0 of 12 evaluable (0 attack, 0 bona fide) | 0 of 12 evaluable (0 attack, 0 bona fide) | 0 of 12 evaluable (0 attack, 0 bona fide) |
| AttackNetV2_1 | 0 of 12 evaluable (0 attack, 0 bona fide) | 0 of 12 evaluable (0 attack, 0 bona fide) | 0 of 12 evaluable (0 attack, 0 bona fide) | 0 of 12 evaluable (0 attack, 0 bona fide) |
| AttackNetV2_2 | 0 of 12 evaluable (0 attack, 0 bona fide) | 0 of 12 evaluable (0 attack, 0 bona fide) | 0 of 12 evaluable (0 attack, 0 bona fide) | 0 of 12 evaluable (0 attack, 0 bona fide) |
| ResNet-18 | 0 of 12 evaluable (0 attack, 0 bona fide) | 0 of 12 evaluable (0 attack, 0 bona fide) | 0 of 12 evaluable (0 attack, 0 bona fide) | 0 of 12 evaluable (0 attack, 0 bona fide) |

Models that decide every dev frame as one class at threshold 0.5 (`public/models.csv`): LivenessNet fold 1 seed 42,
patience 4, ordinary BN: every dev frame attack; LivenessNet fold 4 seed 44, patience 4, ordinary BN: every dev frame
attack.
