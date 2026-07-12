# Time-Locked Pretraining as a Testbed for Machine Origination of Scientific Ideas

## A simulation spec: can an LLM trained only on the pre-backpropagation literature be steered to develop backpropagation?

**Status:** Draft v0.1 — for discussion and preregistration
**Date:** 2026-07-11

---

## 0. Summary

We propose a controlled simulation of scientific discovery: pretrain a language model **from scratch, exclusively on text published before a hard historical cutoff**, and then attempt to **steer/elicit** the model into originating a landmark idea that historically appeared *after* that cutoff. The canonical target is **backpropagation as published by Rumelhart, Hinton & Williams (Nature, 9 October 1986)**, chosen because it is (a) recent enough that a dense machine-readable period corpus exists, (b) a textbook case of *multiple independent discovery* — implying it was derivable from period knowledge — and (c) **machine-verifiable**: a proposed learning rule can simply be implemented and run, so success can be judged by a program with no post-cutoff knowledge in the loop.

The experiment operationalizes what Demis Hassabis has publicly called the "**Einstein test**" for AGI (train to a 1911 cutoff; see whether the system derives general relativity) [H1], in a deliberately more tractable setting, and directly tests the published null hypothesis that LLMs "can't jump" to ideas outside their training distribution [Z1] against the "latent knowledge" result of Tshitoyan et al., who showed that embeddings trained only on pre-year-X literature anticipate post-X discoveries [T1].

Headline design points, each argued from the literature below:

1. **The naive cutoff is wrong.** The full backprop algorithm was publicly circulating from **September 1985** (Rumelhart et al., ICS Report 8506) and mass-distributed from **July 1986** (PDP Vol. 1, Ch. 8) — before the Nature paper. Precursors go back much further (Linnainmaa 1970; Werbos 1974; Parker 1985; LeCun 1985). The canonical cutoff must be **1984-12-31**, inside a lattice of cutoffs (1969 / 1974 / 1984 / 1986-09) that test *different* capabilities, from de-novo synthesis to "awakening a sleeping beauty."
2. **From-scratch pretraining is non-negotiable.** Fine-tuning or "unlearning" a modern model cannot produce a defensible cutoff (effective cutoffs of modern models are provably messy [C2]; fine-tuned models leak anachronisms where period-pretrained models do not [F1]).
3. **No modern generative model may touch the pipeline.** Model-generated data transmits traits through semantically unrelated content ("subliminal learning") [CL1]; even the tokenizer leaks its training distribution [HA1]. Tokenizer, cleaning, and instruction data must all be period-pure.
4. **Steering is a measured dose, not a binary.** Elicitation runs from raw completion prompting through a **preregistered graded hint ladder** (all hints expressible from pre-cutoff sources), reporting discovery rate as a **dose–response curve** — turning "did the experimenters leak the answer?" into a measured quantity.
5. **Execution is the arbiter.** Ideas are scored by implementing the proposed learning rule and testing whether it trains multilayer networks past linear-separability baselines (XOR/parity/encoder — themselves period-canonical problems), because idea-stage novelty judgments systematically over-credit LLMs [S2].
6. **Report a rediscovery *rate*, not a hit.** Following Gould's "replaying the tape" [G1], the claim is statistical: many seeds, prompts, and decodings, with preregistered thresholds — a lone lucky rollout is not a finding.

This experiment is, to our knowledge, unpreempted: the closest existing efforts target physics (GPT-1900/"Machina Mirabilis" [HL1]) or historical worldview (Ranke-4B [R2], talkie-1930 [TK1]), not an algorithmic/CS discovery, and none has the verifiable-reward property that makes backprop uniquely clean.

---

## 1. Motivation and research questions

Whether LLMs can *originate* scientific ideas — rather than retrieve and recombine memorized ones — is contested at the level of first principles. The "stochastic parrot" position holds that LMs stitch together form without meaning [B1]; Chollet argues most benchmark success is crystallized, memorized skill rather than generalization [CH1]; Zahavy's ICML 2026 position paper formalizes discovery as induction + deduction + an abductive "jump," and argues an LLM with a 1905 cutoff could not invent general relativity [Z1]. On the other side: word embeddings trained on time-sliced corpora demonstrably encode *latent, not-yet-published* discoveries [T1]; LLMs beat human experts at predicting unseen experimental outcomes [L1]; and verifier-driven LLM search loops have produced genuinely new mathematics [RP1, N1].

The dispute persists because the natural experiment is confounded: every modern model has already read every idea we might test it on. **Time-locked pretraining removes the confound by construction.** A model whose entire causal history of text ends at time T cannot retrieve an idea first written down after T; anything it produces beyond T is synthesis (or luck, which the rate-based design quantifies).

**Research questions.**

- **RQ1 (Origination):** Can a model pretrained only on pre-T text produce, under period-clean elicitation, the core content of a discovery dated after T? For backprop: differentiable multilayer units + chain-rule credit assignment propagated backward + iterative gradient-descent weight updates + demonstration on nonlinearly-separable tasks.
- **RQ2 (Dose–response):** How much steering — quantified as position on a preregistered hint ladder and as sampling budget — is required before the discovery rate becomes non-negligible? Where does the curve inflect?
- **RQ3 (Synthesis vs. retrieval):** How does the rate change across the cutoff lattice and under *precursor ablations* (removing the optimal-control, autodiff, or Werbos lineages from the corpus)? This separates re-deriving an idea from surfacing a buried document.
- **RQ4 (Inevitability):** Do results support the Merton/Ogburn–Thomas view that discoveries are "in the air" once prerequisites accumulate [M1, O1] — i.e., is backprop-1986 reproducibly derivable from 1984 knowledge, or contingent in Gould's sense [G1]?

Beyond the scientific question, a validated positive result would matter for forecasting (models as instruments for predicting near-future discoveries from today's literature — the live corpus is just another "period corpus"), and a careful null would put empirical teeth into the "can't jump" position at a far more favorable setting than relativity.

---

## 2. Prior art

The proposal sits at the intersection of five literatures. (All citations were independently verified against primary sources; see §12.)

### 2.1 Time-sliced discovery prediction (the founding precedents)

- **Tshitoyan et al. 2019** [T1]: word2vec trained on ~3.3M materials-science abstracts *up to year X* ranked compounds later (post-X) discovered to be thermoelectrics far above baseline — the canonical demonstration that "latent knowledge regarding future discoveries is embedded in past publications," and the template for freeze-corpus-then-score-against-the-future validation.
- **Krenn et al. 2023 (Science4Cast)** [K1]: forecasting future AI research topics as link prediction on a 64k-concept network frozen at year X; competition-grade backtesting of "could the field have gotten there."
- **Sourati & Evans 2023** [SE1]: adding *who studies what* (human accessibility of inferences) improved prediction of future discoveries by up to ~400% over content-only models — reachability is social and cognitive, not just informational. Design lever: steer the model along historically plausible inference paths (optimization, neurophysiology, control theory), not arbitrary ones.
- **BrainBench (Luo et al., Nat. Hum. Behav. 2025)** [L1]: LLMs surpass human neuroscience experts at forced-choice identification of real vs. plausibly-altered experimental results; calibrated confidence tracked accuracy. Supplies a clean forced-choice protocol we adapt for probing (true update rule vs. decoys).
- **Literature-based discovery**: Swanson's fish-oil/Raynaud's inference from disjoint literatures founded the field [SW1], but Moreau's 2023 critique shows the field's standard time-sliced evaluation mostly rewards predicting noisy co-occurrences [MO1] — a direct warning that our evaluation must be a targeted, preregistered rubric on a single landmark with execution-based scoring, not aggregate co-occurrence recovery.

### 2.2 Time-locked / "vintage" LLMs (feasibility)

A fast-moving subfield now trains LMs from scratch on hard-cutoff corpora (curated index: [AV1]):

- **ChronoBERT/ChronoGPT** [HE1] and the instruction-tuned follow-up [HE2]: per-year cutoff model families that match BERT-class baselines and support leakage-free financial backtests — including a demonstration that **instruction tuning itself can be made chronologically consistent**.
- **TiMaGPT** [D1]: peer-reviewed point-in-time GPTs, explicitly "nonprognosticative," with released timestamped datasets.
- **talkie-1930** [TK1]: a **13B** model on **260B tokens** of pre-1931 English, with (a) an instruction-tuned variant bootstrapped from *period reference works* (encyclopedias, letter-writing manuals) and (b) an **architecturally identical modern-corpus twin** (talkie-web-13b-base) as the control — the exact control structure we adopt. Notably, it writes basic Python from in-context examples despite zero exposure to computers — direct evidence of beyond-corpus generalization in a period model.
- **Ranke-4B / History LLMs** [R2]: 4B models with staggered cutoffs (1913–1946) on 80B tokens from a curated 600B-token timestamped corpus; demonstrably hindsight-free (the 1913 model does not know who Hitler is). Their per-document timestamp verification is the curation standard we adopt.
- **GPT-1900 / "Machina Mirabilis"** [HL1]: the closest existing instantiation of *this* experiment — a ~3.3B model on ~22B pre-1900 tokens, aggressively filtered to remove Einstein/quantum leakage, mid-trained on pre-1900 physics, post-trained with SFT + physics-reasoning RL, then prompted with the historical observations that triggered the discoveries. Result: occasional "glimpses" (light as "definite quantities of energy"; gravity–acceleration equivalence) but no clean rediscovery, with the author candid that the model is too small to reason reliably. This sets our prior: expect graded partial signal, size the model bigger, and design the eval for partial credit.
- **TimeCapsuleLLM** [TC1], **MonadGPT** [MG1], **MacBERTh/histLM** [MB1], **TimeLMs** [TL1], and diachronic-pretraining comparisons [F1]: collectively establish (i) what breaks at small scale on noisy OCR corpora (coherence, factuality, tokenization), (ii) that from-scratch period pretraining beats adapting modern models for period fidelity, and (iii) the staggered-cutoff lattice as standard methodology. **Lazaridou et al. 2021** [LZ1] provides time-stratified perplexity as the measure of a model's distance from post-cutoff text, and **Cheng et al. 2024 ("Dated Data")** [C2] proves that *claimed* cutoffs of models built on web crawls diverge badly from *effective* cutoffs — the core argument for provenance-first corpus construction.
- **Owain Evans's "Vintage LLMs"** [E1] is the conceptual manifesto for this paradigm (including the pre-1915 relativity variant); **Hassabis's "Einstein test"** [H1] is its highest-profile framing. The backprop-1986 instantiation appears to be **genuinely open**: no confirmed project targets an algorithmic discovery, and none exploits a verifiable reward.

### 2.3 History of backpropagation (target definition)

The verified timeline that everything in §3–§4 hangs on:

| Date | Event (public written record) |
|---|---|
| 1958/1962 | Rosenblatt: perceptron; *Principles of Neurodynamics* coins "back-propagating error correction" and reports heuristic multilayer error-propagation experiments — over **non-differentiable** units, so no gradient method possible [RO1] |
| 1960 | Widrow & Hoff: LMS/delta rule — exact gradient descent for a single linear unit; extending it past one layer visibly fails (Madaline) [W1] |
| 1960–62 | Kelley; Bryson & Denham; Dreyfus: backward adjoint/gradient methods through staged systems (optimal control); Dreyfus's derivation is pure chain rule [KE1, DR1] |
| 1967 | Amari: SGD training of adaptive classifiers [A1] |
| 1969 | Minsky & Papert, *Perceptrons*: XOR/parity limits; multilayer credit assignment posed as *the* open problem [MP1]; Bryson & Ho textbook makes adjoint methods graduate-standard [KE1] |
| 1970 | Linnainmaa: reverse-mode automatic differentiation (MSc thesis, **in Finnish**; English journal version 1976) [LI1] |
| 1974 | Werbos: full synthesis — reverse-mode gradients for training multilayer perceptron-like models (Harvard PhD thesis; gray literature) [WE1] |
| 1982 | Werbos: first published neural-network application (IFIP proceedings, Springer) [WE1]; Parker files Stanford invention disclosure (unpublished) [P1] |
| Apr 1985 | Parker, "Learning-Logic," MIT TR-47 [P1] |
| Jun 1985 | LeCun, backprop-like rule for asymmetric threshold networks (Cognitiva 85, **in French**) [LC1] |
| Sep 1985 | Rumelhart, Hinton & Williams, ICS Report 8506 — the full paper, as a UCSD tech report [RHW1] |
| 17 Jul 1986 | *PDP* Vol. 1 published; Ch. 8 is the algorithm, mass-distributed [RHW1] |
| 9 Oct 1986 | Nature 323:533–536 [RHW1] |

Two historiographies cross-check this chain — Schmidhuber's (detailed, polemical) [SC1] and Griewank's (numerical-analysis side; notes reverse accumulation was *itself* multiply discovered, e.g. Ostrowski ~1965–71) [GR1].

**Consequence:** backprop is a textbook **Mertonian multiple** (≥4 independent inventors within four years) [M1, O1], and Linnainmaa 1970 / Werbos 1974 are archetypal **sleeping beauties** [V1]. Both properties are exactly what the experiment needs: multiples scholarship implies the idea was entailed by the shared knowledge base (so "the model could have derived it" is a falsifiable, historically calibrated claim), while the sleeping-beauty structure gives us naturally occurring *ablation levers*.

### 2.4 Elicitation and search (how "steering" works)

- **Base models can be steered without instruction tuning:** ~3 constant in-context exemplars recover most instruct-model behavior (URIAL) [LN1]. Our exemplars will be period-authentic formats (review articles, grant proposals) composed only of pre-cutoff text.
- **Alignment narrows distributions:** RLHF measurably reduces output diversity [KI1]; RLVR sharpens but rarely extends a base model's pass@k support (base models win at large k) [Y1]. So the **primary protocol is massive sampling from the untuned base model**, with pass@k as the headline metric; coverage scales log-linearly with samples over four orders of magnitude [BR1], which lets pilot dose–response curves budget the full campaign.
- **Verifier-driven loops discover real things:** FunSearch produced new mathematics by pairing an LLM proposer with a programmatic scorer [RP1]; AlphaEvolve extended this to substantive algorithm discovery [N1]; Coscientist closed the loop through physical experiments [BO1]; R1-Zero showed RL with verifiable rewards elicits reasoning directly from a base model with no SFT [DS1]; Absolute Zero bootstraps via self-play against an executor with zero external data [AZ1]. **Our reward channel — "does the proposed rule train a small multilayer net on XOR/parity/encoder tasks?" — is computable entirely within period knowledge**, structurally unable to leak the answer's *content* (its task choice does encode hindsight; see §8.3 and the verifier-hint accounting in §7.3).
- **Idea-generation studies:** blind expert review found LLM ideas *more novel* than experts' but the advantage evaporates on execution [S1, S2] — hence execution-as-arbiter. Scaled ideation collapses into duplicates [S1], so the search loop needs explicit diversity forcing (high temperature, verbalized-sampling-style enumeration [VS1], island populations [RP1]) with embedding-level dedup as a monitored quantity.
- **Graded hints are measurable:** CHAMP demonstrates concept/hint-annotated evaluation where performance is tracked as a function of supplied side-information [CM1] — the direct model for our hint ladder.

### 2.5 Evaluation methodology and validity

Rubric-tree grading of long-horizon scientific work with a separately validated automated judge (PaperBench / JudgeEval) [PB1]; external human-calibrated yardsticks and contamination auditing (MLE-bench) [ML1]; rediscovery-as-validation done carefully (AI Feynman [AF1]; GNN + symbolic regression re-deriving Newtonian gravity — with candid accounting of the hand-supplied inductive biases that did part of the work [LE1]); memorization-defeating counterfactual task variants (NewtonBench) [NB1]; membership-inference contamination audits (Min-K% Prob, time-split benchmarks) [SH1]; evaluation-awareness and over-elicitation as symmetric threats [NE1, VW1]; and ML preregistration as institutional practice [PR1].

---

## 3. Why backpropagation-1986 is the right first target

1. **Density of the period corpus.** Unlike 1905 physics, the 1950–1984 literature is largely already indexed (MEDLINE from 1946, Science Citation Index to 1900 via backfiles [WS1], OpenAlex metadata [OA1]) and survives in digitized form; the domain-core literature (connectionism, optimization, control, psychology) is small and enumerable (order 10⁵–10⁶ documents).
2. **A smaller "jump."** All conceptual ingredients — differentiable units, gradient descent, chain rule through staged systems, the explicitly posed credit-assignment problem — are in the mainstream record by 1969 (§2.3). Zahavy's argument that discovery requires frame-invention [Z1] is *weakest* here; if models fail even at backprop, that null is maximally informative.
3. **Multiple discovery = calibrated prior.** Four-plus independent inventions in 1974–1986 mean the historical base rate of derivation-given-1974-knowledge is demonstrably nonzero [M1]. No such calibration exists for relativity.
4. **Machine-verifiable success.** A proposed learning rule is a *program*. The verifier (train a 2-3-1 net on XOR; parity; the 4-2-4 encoder) runs with zero post-cutoff knowledge — the property FunSearch-class loops need [RP1] and physics targets lack.
5. **Honest footnote — the recursion irony.** A backprop-trained transformer rediscovering backprop is philosophically piquant but not a methodological confound: the *weights* encode the training corpus's content, not the training algorithm's derivation. The architecture question (does a 2017 transformer + modern optimizer constitute a post-1986 "prior"?) is real and addressed in §8.4: the claim we test is about **knowledge synthesis from a period corpus**, not about re-running history with period tools.

**Success would not mean the model is Rumelhart.** It would mean the 1986 synthesis was latent in the 1984 record and extractable by learned statistical synthesis under measured steering — the Merton hypothesis, made computational. Period-plausible *alternatives* the model might invent instead (weight perturbation, Boltzmann-style stochastic learning, evolutionary search over weights) are recorded and graded as first-class outcomes (§7.2): the general claim is about originating *workable* ideas, not about matching one historical trajectory.

---

## 4. Experimental design

### 4.1 Cutoff lattice (between-model conditions)

Each condition is a separate from-scratch pretraining run on a corpus frozen at the stated date. (Staggered-cutoff lattices are standard practice: TimeLMs [TL1], Ranke-4B [R2].)

| ID | Cutoff | Corpus contains | What it tests |
|---|---|---|---|
| **C-1969** | 1969-12-31 | Perceptrons, LMS, adjoint methods (incl. Bryson & Ho), Amari SGD, Minsky–Papert's freshly posed problem. No reverse-mode AD, no synthesis. | **De novo synthesis.** Hardest, cleanest condition — the idea is genuinely absent; the problem statement is period-native. |
| **C-1974** | 1974-12-31 | + Linnainmaa 1970 (Finnish thesis; include per gray-literature rule), Werbos 1974 (thesis). | Synthesis when the answer exists only in obscure gray literature — the "deep sleeping beauty" condition. |
| **C-1984** ⭐ | 1984-12-31 | + Linnainmaa 1976 (English), Werbos 1982 (published NN application), Dreyfus 1973. Excludes Parker TR-47, LeCun 1985, ICS-8506, PDP. | **Canonical condition.** The historical position of the 1985–86 discoverers: components published but unnoticed. Tests "awakening a sleeping beauty" — the realistic analogue of steering today's models toward tomorrow's ideas. |
| **C-1986.7** | 1986-09-30 | + Parker TR-47, LeCun 1985, ICS-8506, PDP Vol. 1. Excludes only the Nature paper. | Recognition/valuation control: the algorithm is present; does elicitation surface and correctly *rank* it? (This is the condition the naive "just before Hinton published" framing accidentally specifies — kept deliberately, as a control, and as the empirical answer to that framing.) |

**Public-record rule (preregistered):** a document is *in* the corpus for cutoff T iff its date of public deposit/distribution ≤ T and it is machine-readable per §5. Theses count from deposit; unpublished invention disclosures (Parker 1982) do **not** count. Language coverage: English primary; Finnish/French/German/Russian domain-core documents included in original + period-made translations only (no fresh translations — that's a leakage channel). Each inclusion decision for the ~40 known precursor documents (per [SC1, GR1]) is enumerated in an appendix at preregistration time.

### 4.2 Precursor ablations (within C-1984)

Backprop was reached historically from at least three traditions, so ablating each lineage from the corpus estimates which carries the discovery (RQ3):

- **A-ctrl:** remove optimal-control gradient corpus (Kelley, Bryson & Denham, Dreyfus, Bryson & Ho).
- **A-ad:** remove autodiff/numerical-analysis corpus (Linnainmaa, Ostrowski).
- **A-werbos:** remove Werbos 1974/1982 only.
- **A-all:** remove all of the above (approximates C-1969's knowledge state at C-1984's corpus scale — separating *knowledge* effects from *scale* effects).

Ablations are implemented as document-level exclusions plus n-gram/embedding sweeps for quotations and reviews of the ablated works.

### 4.3 Controls

- **Modern twin (positive control):** an architecturally identical, equal-budget model trained on a modern corpus (FineWeb-class), following talkie-web [TK1]. The *entire elicitation harness* must extract backprop from the twin trivially; if it can't, the harness — not the period model — is broken.
- **Decoy targets (sharpshooter guard):** graders and the elicitation team run the identical pipeline against 2–4 preregistered *non-events* (plausible-sounding discoveries that did not happen) and 2–4 *other real* post-cutoff targets (e.g., Boltzmann machine learning, simulated annealing for optimization). Grading toward a known answer is detectable when decoys score high [MO1, AF1].
- **Counterfactual probes (memorization guard):** NewtonBench-style altered-law variants [NB1] — sign-flipped or perturbed update rules — in forced-choice probes (BrainBench protocol [L1]): a model that merely retrieved text fails on counterfactuals; a model that synthesizes should track the *functional* variant.
- **Replay statistics:** every condition runs ≥5 pretraining seeds' worth of elicitation campaigns (elicitation-level seeds; pretraining reruns for the canonical condition only, budget permitting), reporting discovery-rate posteriors rather than existence claims [G1].

---

## 5. Corpus construction

### 5.1 What exists (verified estimates)

- **All scholarly articles ever, by 1986:** Jinha's anchor (cumulative 50M articles by 2009 [J1]) discounted at the measured 3–5%/yr post-1950 growth rate [BM1, PS1] gives **~16–25M articles extant at the cutoff**; at ~4.1k words (~5.5k BPE tokens) per full-text article (S2ORC-measured [S2O]), the journal-article core is **~90–140B tokens**. Books, theses, reports, patents add ~0.25–0.65T → **all scientific/technical text to 1986 ≈ 0.35–0.8T tokens**. The entire scientific record of 1986 fits inside a fraction of one modern pretraining run.
- **Actually machine-readable today:** essentially none of it is in open science corpora (S2ORC full text is ~10% of papers, overwhelmingly post-2000 OA [S2O]); pre-1975 MEDLINE lacks even abstracts [NLM1]. Realistic licensed union (publisher backfiles digitized to first issues, JSTOR, HathiTrust bound serials): **~10–20M pre-1986 articles ≈ 60–120B tokens, nearly all as page images requiring OCR**. Openly redistributable science subset: ~10–30B tokens.
- **Period-general text (for linguistic competence):** HathiTrust's US-public-domain slice ≈ 0.6–0.9T raw tokens [HT1]; American Stories provides ~10¹¹ tokens of pre-1963 newspapers, already extracted and LLM-ready [AS1]; plus Gutenberg, Internet Archive. **Volume is not the binding constraint — OCR quality and rights are.**

### 5.2 Composition

Target mixture (C-1984): ~60% period-general text (books, newspapers, periodicals), ~35% scientific/technical (journals, textbooks, theses, patents, government reports), ~5% **domain core** — the enumerated connectionism/optimization/control/psychology literature, OCR'd at high quality with human QA and oversampled ~4× (curation quality buys reasoning at small scale: Galactica [GA1], phi-1 [PH1]).

### 5.3 Provenance and cutoff hygiene

- **Provenance-first pipeline** (Dolma toolkit [DO1]): every document carries source, edition, and verified publication date; date-unverifiable documents are **excluded**, not guessed.
- Exclude post-cutoff *editions* of pre-cutoff works, reprints with modern forewords, serial volumes spanning the cutoff, and digitization boilerplate (scan headers, library bookplates) [C2, R2].
- **Anachronism canaries:** automated sweeps for post-cutoff coinages and events (for C-1984: "Chernobyl," "perestroika," "Challenger," "backpropagation" as a single token-word, "connectionist" post-1985 senses, "HTML"...), plus embedding-similarity sweeps against a held-out post-cutoff reference set. Note the rubric consequence of Rosenblatt 1962: the *phrase* "back-propagating errors" legitimately predates the cutoff — canaries and graders must key on mechanism, not vocabulary [RO1].
- **Acceptance test on the trained model,** not just the corpus: Dated-Data-style effective-cutoff probing [C2], time-stratified perplexity on post-cutoff text including the Nature paper itself [LZ1], and Min-K% membership inference against known leak candidates [SH1].

### 5.4 OCR without leakage

Pre-1986 scientific text lives in page images, full of math, in pre-digital typography that modern OCR models were not trained on [NG1]. VLM OCR is cheap (~$176/M pages [OL1]) but is a **generative model writing text into the corpus** — a hallucination and (worst-case) leakage channel, given that model-generated data transmits information beyond its visible content [CL1]. Policy:

1. Domain core: classical OCR + human QA (it is small enough).
2. Bulk scientific text: dual-engine transcription (classical + VLM); keep regions where outputs agree above threshold; disagreeing regions go to classical output or human queue — the VLM never contributes text the classical engine didn't independently support.
3. Pilot benchmark (Phase 0): CER and equation-recovery on 1950s–80s journal typesetting (e.g., *Biological Cybernetics* 1975–85), since no published benchmark covers it (open question flagged in the corpus survey; American Stories reports 4–9% CER on 19th-c. newsprint as a floor expectation [AS1]).

### 5.5 Rights

The precedent stack (HathiTrust 2014; Google Books 2015; *Bartz v. Anthropic* 2025) supports transformative research use of **lawfully acquired** in-copyright text and specifically condemns shadow-library sourcing [LG1]. Strategy: library partnerships (HathiTrust research access — whether trained weights are an exportable "derived result" under non-consumptive terms is an open item to negotiate), licensed publisher backfiles, purchased/destructively-scanned copies; NYPL's machine-readable copyright-renewal records promote the ~75% unrenewed majority of 1923–1964 US registrations into the open tier [NY1]. Two-tier release: open weights + corpus manifest for the public-domain tier; gated weights for the full corpus.

---

## 6. Model and training

- **Tokenizer:** BPE trained from scratch on the period corpus only — modern tokenizers measurably encode their training mixture and contain anachronistic merges [HA1].
- **Scale:** sized to data, not ambition. With U ≈ 60–120B unique tokens and the ~4-epoch repetition ceiling (loss ≈ unique-data loss up to ~4 epochs, near-zero value past ~16 [MU1]), effective budget ≈ 240–480B tokens → Chinchilla-consistent [HO1] at **7B parameters (primary)**, 13B stretch. Galactica validates multi-epoch curated-science pretraining at exactly this corpus scale (106B tokens, ~4.25 epochs) [GA1]. Pilots at 0.16B/1.4B (BabyLM-scale results say <1B-token models get grammar but not multi-step reasoning [BB1] — pilots are for pipeline validation, not scientific conclusions; GPT-1900's "too small to reason" experience at 3.3B [HL1] is why the primary model is 7B+).
- **Architecture:** standard modern decoder transformer, unmodified. Anachronistic by design; see §8.4.
- **Stack:** OLMo-2-class open training recipe [O2] with Pythia-style dense checkpoints and fixed data ordering [PY1] — the checkpoints double as an instrument for tracing *when during training* backprop-adjacent concepts (chain rule ↔ perceptron ↔ credit assignment) become linked.
- **Post-training:** none by default (the base model's distributional breadth is the asset [KI1, Y1]). Optional arms below use only period-pure data.
- **Compute/cost anchors:** 6ND for 7B × 300B ≈ 1.3×10²² FLOPs ≈ 12–20k H100-hours ≈ **$30–60k** at market rates (cross-checked against OLMo's disclosed ~39k H100-h/T-token at 7B [O2]); 13B ≈ $60–150k; each pilot <$1k (Karpathy's $672 GPT-2-1.6B reproduction [KA1]). The dominant costs are corpus acquisition, OCR, and audit — not GPUs.

**Hard rules (preregistered):** no modern pretrained initialization; no modern-model-generated text at any stage (OCR exception per §5.4's agreement-gated protocol); no synthetic "textbook" data [PH1] — the generator would embed post-cutoff knowledge [CL1]; modern models permitted only as *deletion-only* filters (never generating), with the residual selection-bias risk documented.

---

## 7. Elicitation and steering protocols

The user-facing question — "can the model be *steered* to develop the idea?" — is operationalized as an escalation of protocols, each preregistered, each with its injected information accounted for.

### 7.1 Protocol ladder

- **E0 — Free completion.** Period-style continuations ("A fundamental difficulty in extending the perceptron convergence procedure to networks with intermediate layers is..."), massive sampling. Establishes the unsteered floor.
- **E1 — URIAL-1985 few-shot.** ~3 constant in-context exemplars in period formats (review article, grant proposal, correspondence), assembled *verbatim or near-verbatim from pre-cutoff texts* [LN1], then problem-conditioned prompts built from period-authentic statements: Minsky–Papert's XOR challenge, Widrow's documented Madaline failure, Rosenblatt's back-propagation aspiration — all quotable from the corpus itself, so the prompt adds no post-cutoff information [MP1, W1, RO1].
- **E2 — Period-pure instruction arm (optional).** SFT data built from period reference works (talkie recipe [TK1]) or fully chronologically consistent instruction tuning [HE2]; run as a comparison arm, expecting diversity loss [KI1].
- **E3 — Verifier-driven search.** FunSearch-style loop [RP1]: the model proposes learning procedures *as runnable code/pseudo-math*; a harness instantiates them on small multilayer nets and scores learning on XOR, parity, and the 4-2-4 encoder (all period-canonical problems [MP1, RHW1]); island populations + diversity maintenance [RP1, S1]; only scalar fitness returns to the loop. Mutation/recombination prompting uses the period model itself; if it proves too weak, a modern model may be used **only as a code-level mutator on candidate programs, never seeing or generating natural language about learning theory** — and this substitution is reported as a separate arm, since it weakens the origination claim.
- **E4 — RLVR on base (optional, R1-Zero pattern [DS1, AZ1]).** Outcome-reward RL against the E3 verifier. Preregistered as a *sampling-efficiency amplifier*, not a capability creator; E1's large-k sampling remains the mandatory control, since RLVR tends to sharpen rather than extend base-model support [Y1].

### 7.2 Metrics

- **Primary: pass@k discovery-rate curves** (unbiased estimator [Y1]) per condition × protocol × hint level, with fitted coverage curves [BR1] used for compute budgeting and extrapolation.
- Semantic-diversity of the proposal pool (embedding dedup rate) as a monitored quantity [S1].
- Outcome taxonomy per successful rollout: (i) backprop-class (chain-rule credit assignment); (ii) workable non-gradient alternative (perturbation/stochastic/evolutionary — successes for RQ1, distinct for RQ4); (iii) rediscovered-precursor (verbatim-ish surfacing of Werbos/Linnainmaa content — retrieval, not synthesis; detected via n-gram/embedding match against the corpus); (iv) non-workable proposals.

### 7.3 The hint ladder (dose–response, CHAMP-style [CM1])

All hints verified expressible from pre-cutoff sources; ladder frozen at preregistration; every reported discovery rate is indexed by ladder level. Illustrative:

| Level | Hint content (period-expressible) |
|---|---|
| L0 | None (E0/E1 problem statement only) |
| L1 | "The obstacle to training intermediate layers deserves attack by the methods of numerical optimization." |
| L2 | "Consider units whose response varies smoothly with their input, so that derivatives exist." |
| L3 | "Gradient methods for staged systems (as in trajectory optimization) compute sensitivities stage by stage, working backward." |
| L4 | "Apply the chain rule to obtain the derivative of the error with respect to *every* connection weight, including those of hidden units." |
| L5 | Near-spoiler: the update-rule form, minus the derivation. |

The deliverable for RQ2 is the **discovery-rate-vs-level curve**; steering "cost" is the area under it. The verifier's task choice (XOR/parity/encoder) is itself scored on this ladder (≈L1-equivalent: it names the problem, not the solution), making the E3 arm's injected information explicit rather than deniable.

### 7.4 Leakage audit of the elicitation channel

Every prompt, exemplar, and hint is (a) sourced to pre-cutoff documents or flagged as experimenter-authored, (b) reviewed by a historian-of-science referee *blind to the target*, tasked with spotting anachronistic concept-structure (e.g., unexplained emphasis on differentiability at L0–L1), and (c) published verbatim with the preregistration. Full sampling budgets are reported — over-elicitation (best-of-N fishing) is the mirror image of sandbagging and is bounded by preregistered budgets [VW1, NE1].

---

## 8. Evaluation

### 8.1 Functional grading (primary, automatic)

A rollout counts as **functional success** iff its proposed procedure, as implemented by the harness (or by two independent human implementers for ambiguous pseudo-math), trains a ≥1-hidden-layer network to criterion on all three canonical tasks, beating (i) no-hidden-layer baselines and (ii) random-search-over-weights at matched compute. This is the Ideation–Execution-Gap lesson institutionalized: no idea is graded on plausibility alone [S2].

### 8.2 Milestone rubric (secondary, human, partial credit)

PaperBench-style decomposition into binary leaves with preassigned weights [PB1], grounded in the verified history (§2.3):

- **M1** — articulates hidden-layer credit assignment as the blocker (in any period framing) [MP1, RO1]
- **M2** — proposes smooth/differentiable units *so that* error derivatives exist (the step Rosenblatt could not take)
- **M3** — derives or states the multilayer chain-rule gradient in any form (Kelley–Bryson–Dreyfus math transferred to networks)
- **M4** — recognizes backward accumulation's efficiency (≈ cost of one forward pass) [LI1, WE1]
- **M5** — full procedure: iterative gradient-descent updates + demonstration/prediction of learned internal representations on XOR/parity/encoder [RHW1]

Grading protocol: ≥3 blinded expert graders per rollout sample; style-normalized transcripts (Si et al. protocol [S1]); interleaved decoy-target and counterfactual-variant items [NB1]; inter-rater reliability (Krippendorff's α) reported; any LLM judge first validated against human labels on a held-out set (JudgeEval pattern [PB1]) — mandatory here because every available judge, human or machine, knows backprop.

### 8.3 Probing arm (forced choice)

BrainBench-protocol probes [L1]: the model chooses (by likelihood) between the true update rule and matched decoys (sign-flipped, wrong-layer-indexed, counterfactually altered [NB1]), across the cutoff lattice. This measures *recognition* separately from *generation* — C-1986.7 should show strong recognition; the interesting question is where recognition first emerges.

### 8.4 Standing objections, answered in-design

| Threat | Mitigation |
|---|---|
| Corpus contamination (a single leaked review invalidates everything) | Provenance-first construction; canaries; effective-cutoff probing; Min-K% audits; published corpus manifest [C2, SH1] |
| Experimenter leakage via prompts (clever Hans) | Hint ladder + blind historian referee + verbatim publication (§7.3–7.4) |
| Over-elicitation / fishing | Preregistered sampling budgets; rate-based claims; decoy targets [VW1, PR1] |
| Hindsight grading (Texas sharpshooter) | Rubric frozen at preregistration; decoys interleaved; IRR reported [MO1, PR1] |
| Memorized-not-synthesized | Precursor ablations; retrieval detection vs. corpus; counterfactual probes [NB1] |
| "The priors did the discovering" (chain rule etc. handed to the model) | Explicit prior inventory published (as in [LE1]); ablation arms withhold specific priors; claims scoped to synthesis-from-stated-priors [CH1] |
| Anachronistic architecture/optimizer | Scoped claim: this tests knowledge synthesis from a period corpus, not a full historical counterfactual. The modern twin control isolates corpus effects from architecture effects [TK1]. Residual philosophical objection acknowledged, not resolved |
| Single-case cherry-picking [MO1] | Secondary target battery (Boltzmann-machine learning, simulated annealing, TD-learning for a C-1984 corpus) run with the same frozen pipeline, reported regardless of outcome |
| Low capability floor makes the test vacuous | Twin control must pass; time-stratified perplexity and standard-reasoning manipulation checks establish the model clears a preregistered competence bar before discovery claims are evaluated [LZ1, BB1] |

### 8.5 Interpretation matrix (preregistered)

- **High rate at L0–L2, C-1984, surviving ablations** → strong evidence for machine origination; Merton operationalized; direct counterexample to [Z1, B1].
- **Success only at C-1984 with Werbos present, collapsing under A-werbos** → models are *sleeping-beauty detectors*: they surface and complete buried literature. Scientifically valuable (this is the deployable capability — today's corpus is full of unread theses), but a different claim.
- **Success only at L4–L5** → models complete near-adjacent syntheses under heavy steering; quantifies the "jump" that remains.
- **Null everywhere, twin passing** → empirical support for the can't-jump position at its most favorable target; publishable as such (preregistration makes the null citable [PR1]).

---

## 9. Phased plan

| Phase | Work | Exit criterion | Est. cost |
|---|---|---|---|
| **P0** (3 mo) | Preregistration draft; domain-core enumeration (~10⁵–10⁶ docs); OCR pilot benchmark on 1950s–80s typesetting; rights negotiations (HathiTrust, backfiles); hint ladder + rubric authored and piloted on human-written period-plausible attempts | OCR CER/equation-recovery targets met; rubric IRR ≥ threshold on pilot; corpus manifest v1 | ~$50–100k (labor-dominated) |
| **P1** (3 mo) | 0.16B/1.4B pilots on C-1984 draft corpus; cutoff acceptance tests; elicitation harness + verifier built and validated on the modern twin at small scale | Twin extraction works; effective-cutoff probes clean | <$10k compute |
| **P2** (4 mo) | 7B C-1984 + 7B modern twin; full elicitation campaign (E0–E3); probing arm | Preregistered analyses run | ~$80–150k |
| **P3** (6 mo) | Cutoff lattice (C-1969, C-1974, C-1986.7) + ablation arms (A-*) at 7B (train-once, elicit-many); optional 13B canonical run; secondary target battery; blinded grading campaign; write-up | — | ~$150–400k depending on arms |

Total: order **$0.3–0.7M** — dominated by corpus/labor, not GPUs. A minimum viable version (P0–P2 only, C-1984 + twin, E0/E1/E3) is ~$150–250k.

---

## 10. Deliverables

1. **Preregistration** (OSF or equivalent): cutoff definitions, corpus manifest hash, prompts, hint ladder, rubric, budgets, analysis plan [PR1].
2. **Corpus artifacts:** open-tier corpus + manifest; audit reports (canaries, effective-cutoff probes, membership inference).
3. **Models:** the cutoff-lattice family with dense checkpoints (a public scientific instrument in the Pythia/Ranke mold [PY1, R2] — useful far beyond this experiment: hindsight-free forecasting baselines, history-of-science counterfactuals, contamination-free evaluation).
4. **Elicitation logs:** every rollout, prompt, and verifier trace.
5. **Findings paper** reporting rate curves, ablation table, probing results, and the interpretation per §8.5 — positive or null.

---

## 11. Open questions carried into P0

1. Can HathiTrust's non-consumptive framework accommodate pretraining with released weights (is a model an exportable "derived result")? [HT1]
2. Measured OCR error/equation-recovery on 1950s–80s typesetting — no published benchmark exists; P0 creates one.
3. Does the ~4-epoch repetition ceiling [MU1] hold on heavily curated scientific text (Galactica's 4.25 epochs suggests yes [GA1])?
4. Where does cross-document conceptual recombination emerge on noisy historical corpora — is 7B enough, given GPT-1900's 3.3B failure [HL1] and talkie's 13B in-context generalization [TK1]?
5. How thin is the base-model distribution's tail on a 100B-token corpus — do the large-k coverage results from trillion-token models [Y1, BR1] transfer?
6. Non-English coverage: Linnainmaa is Finnish, LeCun is French, Ostrowski is Russian/German — does an English-dominant corpus bias which lineages are reachable? [GR1]
7. What rediscovery rate is *meaningful*? Null model needed (e.g., shuffled-domain-core control) to calibrate whether 1-in-10³ rollouts is capability or noise.

---

## 12. References

*Every entry below was independently re-verified against primary or authoritative sources during preparation of this spec; corrections found during verification are incorporated (e.g., BrainBench's journal year is 2025; the 2023 LBD-evaluation critique is single-authored by Moreau).*

**Direct precedents & time-locked LMs**

- [T1] Tshitoyan, V., Dagdelen, J., Weston, L., Dunn, A., Rong, Z., Kononova, O., Persson, K.A., Ceder, G., Jain, A. (2019). Unsupervised word embeddings capture latent knowledge from materials science literature. *Nature* 571, 95–98.
- [K1] Krenn, M., Buffoni, L., Coutinho, B., Eppel, S., Foster, J.G., Gritsevskiy, A., Lee, H., Lu, Y., Moutinho, J.P., et al. (2023). Forecasting the future of artificial intelligence with machine learning-based link prediction in an exponentially growing knowledge network. *Nature Machine Intelligence* 5, 1326–1335.
- [SE1] Sourati, J., Evans, J.A. (2023). Accelerating science with human-aware artificial intelligence. *Nature Human Behaviour* 7, 1682–1696.
- [L1] Luo, X., Rechardt, A., Sun, G., Nejad, K.K., et al., Love, B.C. (2025). Large language models surpass human experts in predicting neuroscience results. *Nature Human Behaviour* 9(2), 305–315.
- [SW1] Swanson, D.R. (1986). Fish oil, Raynaud's syndrome, and undiscovered public knowledge. *Perspectives in Biology and Medicine* 30(1), 7–18.
- [MO1] Moreau, E. (2023). Literature-based discovery: addressing the issue of the subpar evaluation methodology. *Bioinformatics* 39(2), btad090.
- [HE1] He, S., Lv, L., Manela, A., Wu, J. (2025). Chronologically Consistent Large Language Models. arXiv:2502.21206.
- [HE2] He, S., Lv, L., Manela, A., Wu, J. (2025). Chronologically Consistent Generative AI. arXiv:2510.11677.
- [D1] Drinkall, F., Rahimikia, E., Pierrehumbert, J.B., Zohren, S. (2024). Time Machine GPT. *Findings of NAACL 2024*.
- [TK1] Radford, A., Levine, N., Duvenaud, D. (2026). talkie: a 13B vintage language model from 1930. Open-weights release, github.com/talkie-lm/talkie.
- [R2] Göttlich, D., Loibner, D., Jiang, G., Voth, H.-J. (2025–26). History LLMs / Ranke-4B. github.com/DGoettlich/history-llms.
- [HL1] Hla, M. (2026). Machina Mirabilis / GPT-1900. michaelhla.com/blog/machina-mirabilis.html; github.com/michaelhla/gpt1900.
- [TC1] Grigorian, H. (2025). TimeCapsuleLLM. github.com/haykgrigo3/TimeCapsuleLLM.
- [MG1] Langlais, P.-C. (2023). MonadGPT. huggingface.co/Pclanglais/MonadGPT.
- [MB1] Manjavacas, E., Fonteyn, L. (2021/2022). MacBERTh: historically pretrained LM for English (1450–1950); Adapting vs. pre-training LMs for historical languages. *NLP4DH 2021*; *JDMDH* 2022.
- [TL1] Loureiro, D., Barbieri, F., Neves, L., Espinosa Anke, L., Camacho-Collados, J. (2022). TimeLMs: Diachronic Language Models from Twitter. *ACL 2022 demos*.
- [F1] Fittschen, E., Li, S., Lippincott, T., Choshen, L., Messner, C. (2025). Pretraining Language Models for Diachronic Linguistic Change Discovery. arXiv:2504.05523.
- [LZ1] Lazaridou, A., Kuncoro, A., Gribovskaya, E., et al. (2021). Mind the Gap: Assessing Temporal Generalization in Neural Language Models. *NeurIPS 2021*.
- [E1] Evans, O. (2025). Vintage Large Language Models (talk transcript). owainevans.github.io.
- [H1] Hassabis, D. (2026). Remarks proposing the "Einstein test" for AGI (train to 1911; derive general relativity). Public interview, India AI Impact Summit, Feb 2026; multiple press reports.
- [AV1] entanglr (2026). awesome-vintage-llms (curated index). github.com/entanglr/awesome-vintage-llms.
- [Z1] Zahavy, T. (2026). Position: LLMs can't jump. *ICML 2026* position paper; PhilSci-Archive 28024.

**History of backpropagation & discovery scholarship**

- [RHW1] Rumelhart, D.E., Hinton, G.E., Williams, R.J. (1986). Learning representations by back-propagating errors. *Nature* 323, 533–536 (9 Oct 1986); ICS Report 8506 (UCSD, Sept 1985); PDP Vol. 1 Ch. 8 (MIT Press, 17 Jul 1986).
- [LI1] Linnainmaa, S. (1970). MSc thesis, Univ. Helsinki (in Finnish); (1976) Taylor expansion of the accumulated rounding error. *BIT* 16(2), 146–160.
- [WE1] Werbos, P.J. (1974). Beyond Regression. PhD thesis, Harvard; (1982) Applications of advances in nonlinear sensitivity analysis. *Proc. 10th IFIP Conf.*, Springer, 762–770.
- [KE1] Kelley, H.J. (1960). Gradient theory of optimal flight paths. *ARS Journal* 30(10), 947–954; Bryson, A.E., Denham, W.F. (1962). *J. Appl. Mech.* 29, 247–257; Bryson, A.E., Ho, Y.-C. (1969). *Applied Optimal Control*. Blaisdell.
- [DR1] Dreyfus, S.E. (1962). The numerical solution of variational problems. *J. Math. Anal. Appl.* 5(1), 30–45.
- [A1] Amari, S. (1967). A theory of adaptive pattern classifiers. *IEEE Trans. Electronic Computers* EC-16(3), 299–307.
- [P1] Parker, D.B. (1985). Learning-Logic. TR-47, MIT CCREMS; Stanford invention disclosure S81-64 (filed Oct 1982). See Widrow & Lehr, *Proc. IEEE* 78(9), 1990.
- [LC1] LeCun, Y. (1985). Une procédure d'apprentissage pour réseau à seuil asymétrique. *Proc. Cognitiva 85*, Paris, 599–604.
- [RO1] Rosenblatt, F. (1958). The perceptron. *Psych. Review* 65(6), 386–408; (1962) *Principles of Neurodynamics*. Spartan Books.
- [W1] Widrow, B., Hoff, M.E. (1960). Adaptive switching circuits. *IRE WESCON Conv. Record* pt. 4, 96–104.
- [MP1] Minsky, M., Papert, S. (1969). *Perceptrons*. MIT Press.
- [SC1] Schmidhuber, J. (2014–). Who invented backpropagation? people.idsia.ch; (2015) Deep learning in neural networks: an overview. *Neural Networks* 61, 85–117.
- [GR1] Griewank, A. (2012). Who invented the reverse mode of differentiation? *Documenta Mathematica*, Extra Vol. ISMP, 389–400.
- [M1] Merton, R.K. (1961). Singletons and multiples in scientific discovery. *Proc. Am. Phil. Soc.* 105(5), 470–486.
- [O1] Ogburn, W.F., Thomas, D. (1922). Are inventions inevitable? *Political Science Quarterly* 37(1), 83–98.
- [V1] van Raan, A.F.J. (2004). Sleeping Beauties in science. *Scientometrics* 59(3), 467–472; Ke, Q., Ferrara, E., Radicchi, F., Flammini, A. (2015). *PNAS* 112(24), 7426–7431.
- [G1] Gould, S.J. (1989). *Wonderful Life*. W.W. Norton.

**Corpus, rights, OCR**

- [PS1] Price, D.J. de Solla (1963). *Little Science, Big Science*. Columbia Univ. Press.
- [BM1] Bornmann, L., Mutz, R. (2015). Growth rates of modern science. *JASIST* 66(11), 2215–2222; Bornmann, Haunschild & Mutz (2021). *Humanit. Soc. Sci. Commun.* 8, 224.
- [J1] Jinha, A.E. (2010). Article 50 million. *Learned Publishing* 23(3), 258–263.
- [NLM1] NLM. OLDMEDLINE data (1946–1965; no abstracts pre-1975). nlm.nih.gov.
- [WS1] Clarivate. Web of Science Core Collection backfiles / Century of Science (coverage to 1900).
- [OA1] Priem, J., Piwowar, H., Orr, R. (2022). OpenAlex. arXiv:2205.01833.
- [S2O] Lo, K., Wang, L.L., Neumann, M., Kinney, R., Weld, D.S. (2020). S2ORC. *ACL 2020*.
- [HT1] HathiTrust Research Center (2020–21). Extracted Features v2.0/2.5; non-consumptive research framework.
- [AS1] Dell, M., Carlson, J., Bryan, T., Silcock, E., et al. (2023). American Stories. *NeurIPS 2023 D&B*.
- [NG1] Blecher, L., Cucurull, G., Scialom, T., Stojnic, R. (2023). Nougat. arXiv:2308.13418.
- [OL1] Poznanski, J., et al. (2025). olmOCR. arXiv:2502.18443.
- [NY1] NYPL (2019). US Copyright History 1923–1964 (Catalog of Copyright Entries dataset).
- [LG1] Authors Guild v. Google, 804 F.3d 202 (2d Cir. 2015); Authors Guild v. HathiTrust, 755 F.3d 87 (2d Cir. 2014); Bartz v. Anthropic, N.D. Cal. No. 3:24-cv-05417 (2025).

**Training, scaling, contamination**

- [HO1] Hoffmann, J., et al. (2022). Training compute-optimal LLMs (Chinchilla). *NeurIPS 2022*.
- [MU1] Muennighoff, N., Rush, A.M., Barak, B., et al. (2023). Scaling data-constrained language models. *NeurIPS 2023*.
- [GA1] Taylor, R., et al. (2022). Galactica. arXiv:2211.09085.
- [PH1] Gunasekar, S., et al. (2023). Textbooks Are All You Need (phi-1). arXiv:2306.11644.
- [BB1] Hu, M.Y., et al. (2024). Findings of the 2nd BabyLM Challenge. *CoNLL 2024*.
- [PY1] Biderman, S., et al. (2023). Pythia. *ICML 2023*.
- [O2] OLMo Team (2024). 2 OLMo 2 Furious. arXiv:2501.00656.
- [DO1] Soldaini, L., et al. (2024). Dolma. *ACL 2024*.
- [C2] Cheng, J., Marone, M., Weller, O., Lawrie, D., Khashabi, D., Van Durme, B. (2024). Dated Data: tracing knowledge cutoffs in LLMs. *COLM 2024* (Outstanding Paper).
- [CL1] Cloud, A., Le, M., Chua, J., Betley, J., Sztyber-Betley, A., Hilton, J., Marks, S., Evans, O. (2025). Subliminal learning. arXiv:2507.14805.
- [HA1] Hayase, J., Liu, A., Choi, Y., Oh, S., Smith, N.A. (2024). Data mixture inference: what do BPE tokenizers reveal about their training data? *NeurIPS 2024*.
- [KA1] Karpathy, A. (2024). Let's reproduce GPT-2 (1.6B) for $672. github.com/karpathy/llm.c #677.

**Elicitation, search, evaluation**

- [KI1] Kirk, R., et al. (2024). Understanding the effects of RLHF on LLM generalisation and diversity. *ICLR 2024*.
- [LN1] Lin, B.Y., et al. (2024). The unlocking spell on base LLMs (URIAL). *ICLR 2024*.
- [VS1] Zhang, J., et al. (2025). Verbalized sampling. arXiv:2510.01171.
- [RP1] Romera-Paredes, B., et al. (2024). Mathematical discoveries from program search with LLMs (FunSearch). *Nature* 625, 468–475.
- [N1] Novikov, A., et al. (2025). AlphaEvolve. Google DeepMind technical report.
- [LU1] Lu, C., Lu, C., Lange, R.T., Foerster, J., Clune, J., Ha, D. (2024). The AI Scientist. arXiv:2408.06292.
- [BO1] Boiko, D.A., MacKnight, R., Kline, B., Gomes, G. (2023). Autonomous chemical research with LLMs (Coscientist). *Nature* 624, 570–578.
- [DS1] DeepSeek-AI (2025). DeepSeek-R1. arXiv:2501.12948.
- [AZ1] Zhao, A., et al. (2025). Absolute Zero. *NeurIPS 2025*.
- [Y1] Yue, Y., et al. (2025). Does RL really incentivize reasoning capacity in LLMs beyond the base model? *NeurIPS 2025*.
- [S1] Si, C., Yang, D., Hashimoto, T. (2024). Can LLMs generate novel research ideas? *ICLR 2025*.
- [S2] Si, C., Hashimoto, T., Yang, D. (2025). The ideation–execution gap. arXiv:2506.20803.
- [LB1] Ruan, K., et al. (2024/2026). LiveIdeaBench. *Nature Communications*; arXiv:2412.17596.
- [RB1] Liu, Y., et al. (2025). ResearchBench. arXiv:2503.21248; Ke, Y., et al. (2025). BioDisco. arXiv:2508.01285.
- [CM1] Mao, Y., Kim, Y., Zhou, Y. (2024). CHAMP. *Findings of ACL 2024*.
- [BR1] Brown, B., et al. (2024). Large Language Monkeys. arXiv:2407.21787.
- [PB1] Starace, G., et al. (2025). PaperBench. *ICML 2025*.
- [ML1] Chan, J.S., et al. (2024). MLE-bench. *ICLR 2025*.
- [AF1] Udrescu, S.-M., Tegmark, M. (2020). AI Feynman. *Science Advances* 6(16).
- [LE1] Lemos, P., Jeffrey, N., Cranmer, M., Ho, S., Battaglia, P. (2023). Rediscovering orbital mechanics with machine learning. *MLST* 4(4).
- [NB1] Zheng, T., et al. (2025). NewtonBench. arXiv:2510.07172 (ICLR 2026).
- [SH1] Shi, W., et al. (2024). Detecting pretraining data from LLMs (Min-K% Prob). *ICLR 2024*.
- [NE1] Needham, J., et al. (2025). LLMs often know when they are being evaluated. arXiv:2505.23836.
- [VW1] van der Weij, T., et al. (2024). AI sandbagging. arXiv:2406.07358.
- [PR1] Bertinetto, L., Henriques, J.F., et al. (orgs.) (2020–21). NeurIPS pre-registration in ML workshops. *PMLR* v148, v181.
- [FB1] Wang, Z., et al. (2026). FIRE-Bench: evaluating agents on the rediscovery of scientific insights. arXiv:2602.02905.
- [B1] Bender, E.M., Gebru, T., McMillan-Major, A., Mitchell, M. (2021). On the dangers of stochastic parrots. *FAccT 2021*.
- [CH1] Chollet, F. (2019). On the measure of intelligence. arXiv:1911.01547.
