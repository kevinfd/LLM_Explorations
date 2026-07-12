# MVP: the period-clean verifier harness

This is the minimum viable slice of the
[time-locked backprop-rediscovery spec](../specs/backprop-1986-retrodiscovery.md),
built to produce **go/no-go evidence before any corpus or pretraining investment**.

The spec's whole design hangs on one component: a reward channel that can judge
whether a proposed learning rule *works*, using zero post-1986 knowledge
(spec §7.1 E3, §8.1). This MVP builds that component and proves it measures
what it claims to measure — the P1 exit criterion from §9.

## What's here

| file | role |
|---|---|
| `verifier.py` | Sigmoid MLP with metered compute; the three period-canonical tasks (XOR, 3-bit parity — Minsky & Papert 1969; the 4-2-4 encoder); trial runner. Proposers see a `RestrictedNet` that deliberately excludes the gradient helper — they must supply the credit-assignment mechanism themselves. |
| `rules.py` | Positive controls (backprop ± momentum), NewtonBench-style decoys (sign-flipped, shuffled-credit), the period floor (output-layer-only delta rule — the documented Madaline failure), a task-validity baseline (no hidden layer), period-plausible alternatives (weight perturbation, finite differences), and compute-matched random search. |
| `harness.py` | Rule × task × seed matrix with matched budgets, JSONL rollout logs, summary tables. |
| `hints.py` | The preregistered L0–L5 hint ladder (spec §7.3), every hint expressible from pre-1985 sources, plus the period-framed problem statement. |
| `propose_llm.py` | The **modern-twin control** (spec §4.3): prompts a modern Claude model with only the period problem statement at a chosen hint level, sandboxes the returned rule, and scores it through the verifier → dose–response curve. Needs `ANTHROPIC_API_KEY` (or `ant auth login`). |
| `run_validation.py` | The instrument-validation experiment (assertions below). |
| `test_mvp.py` | End-to-end proposer-path test with mock proposals — no API key needed. |

## Reproduce

```bash
pip install numpy
python -m mvp.run_validation          # ~4 min on a laptop CPU
python -m mvp.test_mvp                # proposer path, no API key needed
# with an Anthropic API key:
python -m mvp.propose_llm --sweep --k 5   # modern-twin dose-response curve
```

## Results (10 seeds/cell, 400k-unit budget; full logs in `results/`)

| rule | encoder424 (rate / median units) | parity3 | xor |
|---|---|---|---|
| backprop | 100% / 400 | 100% / 3,360 | 100% / 1,120 |
| backprop_plain | 100% / 3,840 | 100% / 92,960 | 100% / 29,920 |
| decoy_shuffled_credit | 100% / 2,320 | 70% / 130,560 | **10%** / 13,600 |
| decoy_sign_flipped | 0% | 0% | 0% |
| delta_no_hidden | 100% / 80 | 0% | 0% |
| output_only_delta | 100% / 6,280 | 0% | 0% |
| finite_diff | 100% / 44,000 | 40% / 373,920 | 100% / 49,920 |
| weight_perturbation | 100% / 604 | 100% / 4,408 | 100% / 1,244 |
| random_search | 10% / 120,164 | 0% | 100% / 9,684 |

**Verdict: INSTRUMENT VALIDATION PASS** (all 19 preregistered checks; see
`results/validation_summary.md`).

## What this run establishes

1. **The reward channel separates the discovery from its decoys.** Backprop
   passes everything; gradient *ascent* passes nothing; destroying only the
   chain-rule credit routing (shuffled-credit) fails the joint criterion. A
   candidate rule that passes all three tasks is doing real multilayer credit
   assignment — and no single task suffices: XOR's tight 2-3-1 bottleneck is
   the discriminating cell (shuffled-credit and output-only both crack the
   looser tasks via trained-readout-over-random-features).
2. **The efficiency insight is measurable.** Finite differences finds the same
   gradient as backprop but pays ~45× the compute on XOR (49,920 vs 1,120
   units) — milestone M4 of the spec's rubric (reverse accumulation ≈ one
   forward pass) shows up directly in the meter.
3. **Period-plausible alternatives are real.** Weight perturbation solves all
   three tasks at toy scale. Functional success is therefore *necessary but
   not sufficient* for "backprop-class": the outcome taxonomy (§7.2) and
   milestone rubric (§8.2) classify the mechanism; the verifier certifies
   workability. Both layers are needed, exactly as the spec argues.
4. **The proposer pipeline works end-to-end.** A mock proposal implementing
   chain-rule learning against the restricted interface (no gradient helper)
   achieves functional success; plausible-but-wrong and broken proposals fail
   (`test_mvp.py`).

## What's next (in order of cost)

1. **Modern-twin dose–response** (`propose_llm.py --sweep`, ~dollars of API):
   a modern model should reach functional success from the L0 period prompt —
   validating elicitation end-to-end and producing the first hint-ladder curve.
2. **Pilot time-locked model** (spec §9 P1, <$10k): 0.16B/1.4B pretraining on a
   draft pre-1985 corpus slice, cutoff acceptance tests, then point *this*
   harness at it.
3. **Full C-1984 + twin runs** (spec §9 P2).

## Known limitations

- The sandbox restricts builtins and imports (numpy/math only) but is **not**
  a security boundary; run untrusted proposals in an isolated environment.
- Budgets, thresholds, and hyperparameters here are the *pilot* settings; the
  real experiment freezes them at preregistration after this kind of pilot.
- The verifier's task choice encodes hindsight (~L1 on the hint ladder), which
  the spec accounts for explicitly (§7.3).
