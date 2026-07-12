# LLM Explorations

Research specs and experiments around large language models.

## Contents

- [`mvp/`](mvp/) — **Runnable MVP of the spec's verifier harness.** The
  period-clean reward channel (propose a learning rule → it gets implemented and
  compute-metered on XOR/parity/encoder), validated against positive controls,
  decoys, baselines, and period-plausible alternatives — plus the L0–L5 hint
  ladder and a modern-model "twin control" proposer. Instrument validation:
  **PASS** (see `mvp/README.md`).
- [`specs/backprop-1986-retrodiscovery.md`](specs/backprop-1986-retrodiscovery.md) —
  **Time-Locked Pretraining as a Testbed for Machine Origination of Scientific Ideas.**
  A literature-backed spec for a simulation that pretrains an LLM from scratch on text
  published only *before* a hard historical cutoff (canonically 1984-12-31, just before the
  1985–86 backpropagation publications), then measures whether the model can be steered to
  originate backpropagation-like ideas from period-available knowledge — with a cutoff
  lattice, precursor ablations, a preregistered graded hint ladder, a period-clean
  verifier-driven search loop, and execution-based grading. Every citation in the spec was
  independently verified against primary sources.
