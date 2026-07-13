# Verifier instrument validation

Seeds per cell: 10 - Budget: 400,000 units - Verdict: **PASS**

| rule | encoder424 (rate / median units) | parity3 (rate / median units) | xor (rate / median units) |
|---|---|---|---|
| backprop | 100% / 400 | 100% / 3,360 | 100% / 1,120 |
| backprop_plain | 100% / 3,840 | 100% / 92,960 | 100% / 29,920 |
| decoy_shuffled_credit | 100% / 2,320 | 70% / 130,560 | 10% / 13,600 |
| decoy_sign_flipped | 0% / -- | 0% / -- | 0% / -- |
| delta_no_hidden | 100% / 80 | 0% / -- | 0% / -- |
| finite_diff | 100% / 44,000 | 40% / 373,920 | 100% / 49,920 |
| output_only_delta | 100% / 6,280 | 0% / -- | 0% / -- |
| random_search | 10% / 120,164 | 0% / -- | 100% / 9,684 |
| weight_perturbation | 100% / 604 | 100% / 4,408 | 100% / 1,244 |

## Checks

- PASS  backprop on xor: solve-rate 100% (want >= 70%)
- PASS  decoy_sign_flipped on xor: solve-rate 0% (want <= 10%)
- PASS  backprop on parity3: solve-rate 100% (want >= 70%)
- PASS  decoy_sign_flipped on parity3: solve-rate 0% (want <= 10%)
- PASS  backprop on encoder424: solve-rate 100% (want >= 70%)
- PASS  decoy_sign_flipped on encoder424: solve-rate 0% (want <= 10%)
- PASS  decoy_shuffled_credit on xor: solve-rate 10% (want <= 10%)
- PASS  delta_no_hidden on xor: solve-rate 0% (want <= 0%)
- PASS  delta_no_hidden on parity3: solve-rate 0% (want <= 0%)
- PASS  joint functional-success(backprop) == True
- PASS  joint functional-success(backprop_plain) == True
- PASS  joint functional-success(decoy_sign_flipped) == False
- PASS  joint functional-success(decoy_shuffled_credit) == False
- PASS  joint functional-success(output_only_delta) == False
- PASS  joint functional-success(delta_no_hidden) == False
- PASS  joint functional-success(random_search) == False
- PASS  backprop beats random_search on xor (100% vs 100%)
- PASS  backprop beats random_search on parity3 (100% vs 0%)
- PASS  backprop beats random_search on encoder424 (100% vs 10%)
