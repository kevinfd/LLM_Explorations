# LLM Explorations

Research specs and experiments around large language models.

## Contents

- [`pretrain/`](pretrain/) — **Runnable time-locked pretraining pipeline.** The
  spec's Phase-P1 slice: provenance-first corpus → hard cutoff filter +
  anachronism-canary audit → from-scratch BPE tokenizer → from-scratch tiny
  transformer with dense checkpoints → cutoff-integrity acceptance probes. Runs
  on a laptop CPU against real public-domain text (1599–1920, cutoff 1900).
  **Pipeline PASS**: the hard cutoff guarantees hold, and — once the model is
  trained enough — the behavioral era signal separates cleanly (pre-cutoff
  entity surprisal 5.2 vs post-cutoff 7.5 nats), tracing the very scale
  threshold the spec predicts (see `pretrain/README.md`).
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
