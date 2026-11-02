# Annex: sensitivity of the M1 result to the 16 test subjects (N1 part 1, run `20261024-N1-subjects-a5fcbbc`)

Existing scores only; one subject at a time excluded from the computation; thresholds unchanged. Source: `n1-nuaa-stability/tables/subject_sensitivity/n1_subject_sensitivity_public.json`.

| model | M1 threshold | ACER@0.5 | ACER@M1 | ΔACER (M1 − 0.5) | leave-one-subject-out ΔACER min | max | sign changes | NA |
|---|---|---|---|---|---|---|---|---|
| LivenessNet | 0.181055 | 0.443333 | 0.410000 | -0.033333 | -0.051750 | -0.009568 | 0/16 | 0 |
| AttackNet V1 | 0.329237 | 0.273333 | 0.270000 | -0.003333 | -0.010165 | 0.002542 | 2/16 | 0 |
| AttackNet V2.1 | 0.870329 | 0.393333 | 0.340000 | -0.053333 | -0.067033 | -0.019845 | 0/16 | 0 |
| AttackNet V2.2 | 0.696933 | 0.466667 | 0.500000 | 0.033333 | 0.019910 | 0.048762 | 0/16 | 0 |
