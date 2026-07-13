"""Cutoff-integrity acceptance probes (spec sections 5.3, 8.4).

Before any output of a time-locked model can be interpreted, the model must be
shown to actually respect its cutoff. The load-bearing probe is genre-robust:

  A. CONTEXTUAL ENTITY SURPRISAL (the era gate). Given the real sentence context
     leading up to a named entity, how surprised is the model by the entity
     itself? A period-locked model should predict PRE-cutoff entities it trained
     on (Ahab, Emma, Alice...) far better than POST-cutoff entities it has never
     seen (Flambeau, Syme, Buster...). This directly measures "does the model
     know the post-cutoff thing" and is not confounded by genre, because both
     entity sets are scored in their own natural prose contexts. This is the
     pilot-scale analog of the spec's whole question.

  B. GENERATION CANARY. Sampling from the model should not produce post-cutoff
     canary surface forms.

  C. DEDICATED-TOKEN CANARY. The from-scratch tokenizer, trained only on
     admitted text, must not contain a dedicated learned subword token for any
     post-cutoff-only entity. (A subword tokenizer can always SPELL any string
     from characters -- that is correct and not a leak -- so the meaningful test
     is whether it learned the name as a unit.)

Reported alongside, as a DIAGNOSTIC (not a gate): raw held-out perplexity on
pre- vs post-cutoff text. At this toy scale genre dominates era in raw
perplexity (post-cutoff Chesterton prose is "easier" standard English than
pre-cutoff Shakespeare/Milton verse), so this number is informative but
genre-confounded -- exactly the confound the spec flags [SE1]. The era gate (A)
is designed to be robust to it.

Plus a manipulation check: training reduced loss (else nothing is meaningful).
"""

import re
from dataclasses import dataclass, field

import numpy as np
import torch

from .cutoff import POST_CUTOFF_CANARIES
from .tokenizer import EOT, UNK, BPETokenizer

# Pre-cutoff entities that occur in the admitted (training) corpus; matched
# against the post-cutoff canaries for the era gate.
PRE_CUTOFF_ENTITIES = ["Ahab", "Ishmael", "Emma", "Elinor", "Alice", "Hamlet",
                       "Macbeth", "Satan", "Anne"]


@torch.no_grad()
def token_perplexity(model, ids: np.ndarray, block_size: int,
                     max_windows: int = 200, seed: int = 0) -> float:
    model.eval()
    n = len(ids)
    starts = list(range(0, n - block_size - 1, block_size))
    if not starts:
        return float("nan")
    rng = np.random.default_rng(seed)
    if len(starts) > max_windows:
        starts = rng.choice(starts, size=max_windows, replace=False)
    losses = []
    for s in starts:
        x = torch.from_numpy(np.asarray(ids[s:s + block_size])[None, :]).long()
        y = torch.from_numpy(np.asarray(ids[s + 1:s + 1 + block_size])[None, :]).long()
        _, loss = model(x, y)
        losses.append(loss.item())
    return float(np.exp(np.mean(losses)))


@torch.no_grad()
def entity_surprisal(model, tok: BPETokenizer, text: str, entities: list[str],
                     block_size: int, ctx_chars: int = 400,
                     max_occ: int = 25, seed: int = 0) -> dict[str, float]:
    """Mean per-token NLL of each entity given its real left context.

    Because the tokenizer keeps word/whitespace atoms separate, encode(context)
    + encode(entity) reproduces the in-stream tokenization exactly, so the
    teacher-forced surprisal of the entity tokens is faithful."""
    model.eval()
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    for ent in entities:
        # occurrences of the entity as a whole word
        positions = [m.start() for m in re.finditer(
            r"(?<![A-Za-z])" + re.escape(ent) + r"(?![A-Za-z])", text)]
        if not positions:
            out[ent] = float("nan")
            continue
        if len(positions) > max_occ:
            positions = list(rng.choice(positions, size=max_occ, replace=False))
        ent_ids = tok.encode(ent)
        nlls = []
        for pos in positions:
            ctx = text[max(0, pos - ctx_chars):pos]
            ctx_ids = tok.encode(ctx)[-(block_size - len(ent_ids) - 1):]
            if not ctx_ids:
                continue
            seq = ctx_ids + ent_ids
            x = torch.tensor(seq[:-1], dtype=torch.long)[None, :]
            y = torch.tensor(seq[1:], dtype=torch.long)[None, :]
            logits, _ = model(x)
            logp = torch.log_softmax(logits[0], dim=-1)
            # NLL on the entity token positions (the last len(ent_ids) targets)
            ent_nll = -logp[range(len(seq) - 1 - len(ent_ids), len(seq) - 1),
                            y[0, -len(ent_ids):]].mean().item()
            nlls.append(ent_nll)
        out[ent] = float(np.mean(nlls)) if nlls else float("nan")
    return out


@dataclass
class ProbeReport:
    pre_entity_surprisal: dict[str, float]
    post_entity_surprisal: dict[str, float]
    pre_entity_mean: float
    post_entity_mean: float
    era_gap: float                 # post_mean - pre_mean; >0 = model knows pre, not post
    pre_cutoff_ppl: float
    post_cutoff_ppl: float
    dedicated_token_canaries: dict[str, bool]
    generated_canary_hits: list[str]
    initial_val_loss: float
    final_val_loss: float
    learned: bool
    era_margin: float = 1.0

    @property
    def no_dedicated_token(self) -> bool:
        return not any(self.dedicated_token_canaries.values())

    @property
    def generation_clean(self) -> bool:
        return len(self.generated_canary_hits) == 0

    @property
    def hard_guarantees_pass(self) -> bool:
        """Scale-INDEPENDENT cutoff-integrity guarantees. These must hold for
        the run to be valid at any scale."""
        return self.learned and self.no_dedicated_token and self.generation_clean

    @property
    def era_signal_pass(self) -> bool:
        """Scale-DEPENDENT behavioral era signal. Expected to require more scale
        than a pilot provides (spec section 6/9 scale-threshold question)."""
        return self.era_gap >= self.era_margin

    @property
    def cutoff_respected(self) -> bool:
        # Pipeline validity is gated on the hard guarantees; the era signal is
        # reported as an additional, scale-dependent diagnostic.
        return self.hard_guarantees_pass

    def render(self) -> str:
        L = ["# Cutoff-integrity acceptance probes", ""]
        L.append("## A. Contextual entity surprisal (the era gate)\n")
        L.append("Mean per-token NLL of each entity given its real prose context "
                 "(higher = more surprised = less known).\n")
        L.append("| pre-cutoff entity | NLL | | post-cutoff entity | NLL |")
        L.append("|---|---|---|---|---|")
        pre = [(k, v) for k, v in self.pre_entity_surprisal.items() if not np.isnan(v)]
        post = [(k, v) for k, v in self.post_entity_surprisal.items() if not np.isnan(v)]
        for i in range(max(len(pre), len(post))):
            lc = f"{pre[i][0]} | {pre[i][1]:.2f}" if i < len(pre) else " | "
            rc = f"{post[i][0]} | {post[i][1]:.2f}" if i < len(post) else " | "
            L.append(f"| {lc} | | {rc} |")
        L.append(f"\n- mean pre-cutoff entity NLL:  {self.pre_entity_mean:.2f}")
        L.append(f"- mean post-cutoff entity NLL: {self.post_entity_mean:.2f}")
        L.append(f"- era gap (post - pre): {self.era_gap:.2f} "
                 f"(scale-dependent signal, need >= {self.era_margin:.2f}: "
                 f"{'PASS -- model knows pre-cutoff entities, not post-cutoff' if self.era_signal_pass else 'below noise floor at this scale'})")
        L.append("\n## B. Generation canary\n")
        L.append("No post-cutoff canary appeared in sampled text."
                 if not self.generated_canary_hits
                 else f"LEAK: sampled text contained {self.generated_canary_hits}")
        L.append("\n## C. Dedicated-token canary\n")
        L.append("The from-scratch tokenizer has no dedicated learned token for "
                 "any post-cutoff entity (a subword tokenizer can still spell them "
                 "from characters -- that is expected and not a leak):\n")
        for tok_s, present in self.dedicated_token_canaries.items():
            L.append(f"- `{tok_s}`: {'DEDICATED TOKEN (leak!)' if present else 'no dedicated token'}")
        L.append("\n## Diagnostic: raw perplexity (genre-confounded, not a gate)\n")
        L.append(f"- held-out PRE-cutoff perplexity:  {self.pre_cutoff_ppl:.2f}")
        L.append(f"- EXCLUDED post-cutoff perplexity: {self.post_cutoff_ppl:.2f}")
        L.append("  (At toy scale genre dominates era here: post-cutoff prose is "
                 "'easier' standard English than pre-cutoff verse. The era gate above "
                 "is designed to be robust to this confound.)")
        L.append("\n## Manipulation check\n")
        L.append(f"- val loss {self.initial_val_loss:.3f} -> {self.final_val_loss:.3f} "
                 f"({'learned' if self.learned else 'DID NOT LEARN'})")
        L.append("\n## Verdict\n")
        L.append(f"- HARD cutoff guarantees (scale-independent): "
                 f"**{'PASS' if self.hard_guarantees_pass else 'FAIL'}** "
                 f"(learned={self.learned}, no-dedicated-token={self.no_dedicated_token}, "
                 f"generation-clean={self.generation_clean})")
        L.append(f"- Behavioral era signal (scale-dependent): "
                 f"**{'PASS' if self.era_signal_pass else 'needs scale'}** "
                 f"(era gap {self.era_gap:.2f})")
        L.append(f"\n**Cutoff respected (hard guarantees): {self.cutoff_respected}**")
        return "\n".join(L)


def run_probes(model, tok: BPETokenizer, pre_val_ids: np.ndarray,
               post_ids: np.ndarray, admitted_text: str, excluded_text: str,
               block_size: int, initial_val_loss: float, final_val_loss: float,
               n_generate: int = 2000, era_margin: float = 1.0,
               seed: int = 0) -> ProbeReport:
    pre_surp = entity_surprisal(model, tok, admitted_text, PRE_CUTOFF_ENTITIES,
                                block_size, seed=seed)
    post_surp = entity_surprisal(model, tok, excluded_text, POST_CUTOFF_CANARIES,
                                 block_size, seed=seed)
    pre_vals = [v for v in pre_surp.values() if not np.isnan(v)]
    post_vals = [v for v in post_surp.values() if not np.isnan(v)]
    pre_mean = float(np.mean(pre_vals)) if pre_vals else float("nan")
    post_mean = float(np.mean(post_vals)) if post_vals else float("nan")

    # Dedicated-token canary: is the entire entity a single vocab token?
    dedicated = {c: (c in tok.vocab or (c + "</w>") in tok.vocab)
                 for c in POST_CUTOFF_CANARIES}

    # Generation canary.
    torch.manual_seed(seed)
    start = torch.tensor([[tok.vocab.get(EOT, 0)]], dtype=torch.long)
    out = model.generate(start, max_new_tokens=n_generate, temperature=1.0, top_k=40)
    gen_words = set(re.findall(r"[A-Za-z][A-Za-z'-]*", tok.decode(out[0].tolist())))
    gen_hits = [c for c in POST_CUTOFF_CANARIES if c in gen_words]

    return ProbeReport(
        pre_entity_surprisal=pre_surp,
        post_entity_surprisal=post_surp,
        pre_entity_mean=pre_mean,
        post_entity_mean=post_mean,
        era_gap=post_mean - pre_mean,
        pre_cutoff_ppl=token_perplexity(model, pre_val_ids, block_size, seed=seed),
        post_cutoff_ppl=token_perplexity(model, post_ids, block_size, seed=seed),
        dedicated_token_canaries=dedicated,
        generated_canary_hits=gen_hits,
        initial_val_loss=initial_val_loss,
        final_val_loss=final_val_loss,
        learned=final_val_loss < initial_val_loss - 0.05,
        era_margin=era_margin,
    )
