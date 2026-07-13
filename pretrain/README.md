# Time-locked pretraining pilot

The second runnable slice of the
[spec](../specs/backprop-1986-retrodiscovery.md): an end-to-end **time-locked
pretraining pipeline** with the cutoff-integrity acceptance tests the spec's
Phase P1 requires (§9) before any discovery claim is meaningful. It exercises
the exact machinery of §5 (provenance-first corpus, hard cutoff, canary audit),
§6 (from-scratch tokenizer, from-scratch tiny transformer, dense checkpoints,
reproducible data order), and §8.4 (effective-cutoff probes) — at a scale that
runs on a laptop CPU in a few minutes.

## Why this corpus

The environment's egress policy reaches exactly one usable text source, so the
pilot corpus is the **NLTK Gutenberg sample**: 18 real public-domain works
spanning **1599–1920**, each carrying a `[Title by Author YEAR]` provenance
header. This is a faithful stand-in for the pre-1986 corpus because it has the
one property the pipeline needs to be non-trivial: a cutoff (1900) that
**genuinely partitions** the data — 13 works admitted, 5 (Chesterton 1908–11,
Bryant 1918, Burgess 1920) excluded — and named entities unique to the excluded
works that serve as real anachronism canaries.

## Pipeline stages

| stage | module | spec |
|---|---|---|
| 1. Provenance-first corpus | `corpus.py` | §5.3 — authoritative publication year per doc, cross-checked against the in-file header; date-unverifiable docs excluded, never guessed; digitization boilerplate stripped |
| 2. Hard cutoff + canary audit | `cutoff.py` | §5.3, §8.4 — metadata gate (authoritative year ≤ cutoff) **and** content gate (no post-cutoff entity survives in the admitted corpus) |
| 3. From-scratch BPE tokenizer | `tokenizer.py` | ref [HA1] — trained only on the admitted corpus (a modern tokenizer leaks its training distribution and anachronistic vocabulary) |
| 4. Pretraining | `model.py`, `train.py` | §6 — from-scratch decoder transformer, fixed data order, dense checkpoints [PY1] |
| 5. Acceptance probes | `probe.py` | §5.3, §8.4 — see below |

## Acceptance probes (spec §8.4)

The spec separates cutoff guarantees that hold **at any scale** from a
behavioral era signal that **needs scale**. The pipeline reports both:

- **Hard guarantees (scale-independent) — the validity gate.**
  1. *Provenance gate*: post-cutoff docs excluded by verified date (enforced at
     corpus assembly; the run aborts if the content canary finds a leak).
  2. *Dedicated-token canary*: the from-scratch tokenizer has no learned subword
     for any post-cutoff entity. (A subword tokenizer can always *spell* any
     string from characters — that is correct, not a leak — so the meaningful
     test is whether it learned the name as a *unit*.)
  3. *Generation canary*: sampling never emits a post-cutoff entity.
- **Behavioral era signal (scale-dependent) — reported, not gated.**
  *Contextual entity surprisal*: given the real prose context leading up to a
  name, the model should predict PRE-cutoff entities it trained on far better
  than POST-cutoff entities it has never seen. This is the genre-robust,
  pilot-scale analog of the spec's whole question ("does the model know the
  post-cutoff thing?").
- **Diagnostic (genre-confounded) — reported only.**
  Raw held-out perplexity, pre- vs post-cutoff. At toy scale genre dominates era
  here (post-cutoff prose is "easier" standard English than pre-cutoff verse) —
  the exact confound the spec flags [SE1]; the era gate above is built to be
  robust to it.

## Reproduce

```bash
pip install torch numpy
python -m pretrain.run_pipeline                      # 1900 cutoff, ~4–5 min
python -m pretrain.run_pipeline --cutoff 1850        # a different partition
python -m pretrain.cutoff                            # just the audit
```

Artifacts land in `results/pretrain/`: `cutoff_audit.md`, `tokenizer.json`,
dense `checkpoints/`, `probe_report.md`, `summary.json`.

## What this establishes (full run: 3.2M params, vocab 3000, 3000 steps, 1900 cutoff)

1. **The cutoff machinery works on real data.** The provenance gate + content
   canary certify the admitted corpus period-pure; the tokenizer and generation
   canaries confirm post-cutoff entities are neither learned as units nor
   emitted. These are the hard guarantees the whole experiment rests on, and
   they hold. **PASS.**
2. **The pipeline is a real from-scratch stack.** A BPE tokenizer and a decoder
   transformer are trained from zero on the filtered corpus (val loss 8.17 →
   2.72), with fixed data order and dense checkpoints — the reproducibility
   posture of §6.
3. **The behavioral era signal appears once the model is trained enough — and
   its scale-dependence is itself the finding the spec predicts.**
   - At **1.8M params / 600 steps** (well under one epoch), entity prediction
     did *not* separate eras (gap ≈ −0.2): the model hadn't memorized even its
     *training* entities — the BabyLM / "too small to reason" regime ([BB1];
     GPT-1900).
   - At **3.2M params / 3000 steps / dropout 0** (≈2 epochs), it separates
     cleanly: mean pre-cutoff entity surprisal **5.20** vs post-cutoff **7.55**
     nats (**era gap 2.35**, margin 1.0 → PASS). Every pre-cutoff entity (Ahab
     2.4, Alice 3.6, Ishmael 3.4) is predicted better than every post-cutoff one
     (MacIan 9.7, Turnbull 8.9, Gregory 8.5). Raw perplexity agrees once genre
     is matched (pre **15.5** < post **21.2**).

   So the pilot both validates the instrument *and* traces the scale threshold
   at which "does the model know the post-cutoff thing?" becomes measurable —
   the core question the full scale-up (§9 P2/P3) is built to answer.

## Scaling to the real experiment

Swap the pilot corpus for the provenance-tagged pre-1986 scientific + general
corpus (§5); keep every gate. The tokenizer, training loop, checkpointing, and
all three acceptance probes transfer unchanged — only scale (corpus size, model
size, epochs) and the canary/entity lists change. The modern-corpus **twin
control** (§4.3) is the same pipeline run on a modern corpus.

## Known limitations

- Pilot scale: conclusions are about the *pipeline*, not about what a period
  model knows. The era signal needs the real scale-up.
- One reachable data source (egress policy); the real corpus work (OCR, rights,
  provenance at scale) is the spec's §5 / §9 P0.
- The 1900/entity canaries stand in for §5.3's post-1986 coinage canaries.
