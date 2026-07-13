# Cutoff-integrity acceptance probes

## A. Contextual entity surprisal (the era gate)

Mean per-token NLL of each entity given its real prose context (higher = more surprised = less known).

| pre-cutoff entity | NLL | | post-cutoff entity | NLL |
|---|---|---|---|---|
| Ahab | 2.39 | | Buster | 5.31 |
| Ishmael | 3.43 | | Chesterton | 6.39 |
| Emma | 4.60 | | Syme | 6.96 |
| Elinor | 4.19 | | MacIan | 9.72 |
| Alice | 3.58 | | Turnbull | 8.89 |
| Hamlet | 7.21 | | Flambeau | 7.04 |
| Macbeth | 8.05 | | Gregory | 8.51 |
| Satan | 8.29 | |  |  |
| Anne | 5.03 | |  |  |

- mean pre-cutoff entity NLL:  5.20
- mean post-cutoff entity NLL: 7.55
- era gap (post - pre): 2.35 (scale-dependent signal, need >= 1.00: PASS -- model knows pre-cutoff entities, not post-cutoff)

## B. Generation canary

No post-cutoff canary appeared in sampled text.

## C. Dedicated-token canary

The from-scratch tokenizer has no dedicated learned token for any post-cutoff entity (a subword tokenizer can still spell them from characters -- that is expected and not a leak):

- `Buster`: no dedicated token
- `Chesterton`: no dedicated token
- `Syme`: no dedicated token
- `MacIan`: no dedicated token
- `Turnbull`: no dedicated token
- `Flambeau`: no dedicated token
- `Valentin`: no dedicated token
- `Gregory`: no dedicated token

## Diagnostic: raw perplexity (genre-confounded, not a gate)

- held-out PRE-cutoff perplexity:  15.47
- EXCLUDED post-cutoff perplexity: 21.18
  (At toy scale genre dominates era here: post-cutoff prose is 'easier' standard English than pre-cutoff verse. The era gate above is designed to be robust to this confound.)

## Manipulation check

- val loss 8.171 -> 2.716 (learned)

## Verdict

- HARD cutoff guarantees (scale-independent): **PASS** (learned=True, no-dedicated-token=True, generation-clean=True)
- Behavioral era signal (scale-dependent): **PASS** (era gap 2.35)

**Cutoff respected (hard guarantees): True**