"""From-scratch byte-pair-encoding tokenizer (spec requirement, ref [HA1]).

A modern tokenizer's merge list encodes its training distribution and carries
anachronistic vocabulary, so the experiment must train the tokenizer *only* on
the period corpus. This is a self-contained BPE implementation (no external
tokenizer library) trained on the admitted, cutoff-filtered text.

Word-level BPE with incremental pair-count updates (the standard fast variant):
train time is a few seconds on the pilot corpus.
"""

import json
import re
from collections import Counter, defaultdict

END = "</w>"                      # end-of-word marker
UNK = "<|unk|>"
EOT = "<|endoftext|>"
SPECIALS = [EOT, UNK]

_WORD_RE = re.compile(r"<\|endoftext\|>|[A-Za-z]+|[0-9]|\s+|[^\sA-Za-z0-9]")


def _pre_tokenize(text: str) -> list[str]:
    """Split into atoms: words, single digits, whitespace runs, punctuation,
    and the literal EOT token. Whitespace is preserved as its own atom so the
    tokenizer is reversible."""
    return _WORD_RE.findall(text)


class BPETokenizer:
    def __init__(self):
        self.merges: dict[tuple[str, str], int] = {}   # pair -> rank
        self.vocab: dict[str, int] = {}                # token string -> id
        self.inv_vocab: dict[int, str] = {}

    # ---- training -------------------------------------------------------
    def train(self, text: str, vocab_size: int = 4000, min_freq: int = 2,
              verbose: bool = False) -> "BPETokenizer":
        atoms = _pre_tokenize(text)
        word_freq: Counter = Counter(a for a in atoms if a != EOT)

        # Each word -> list of symbols (chars) + end marker; whitespace/punct
        # atoms stay atomic (wrapped so merges don't cross atom boundaries).
        words: dict[str, list[str]] = {}
        for w in word_freq:
            if w.isalpha():
                words[w] = list(w) + [END]
            else:
                words[w] = [w]  # keep whitespace/punct/digit atoms whole

        base_chars = set()
        for syms in words.values():
            base_chars.update(syms)

        # Pair statistics across the whole word set (weighted by word freq).
        def get_pair_stats() -> Counter:
            stats: Counter = Counter()
            for w, syms in words.items():
                f = word_freq[w]
                for i in range(len(syms) - 1):
                    stats[(syms[i], syms[i + 1])] += f
            return stats

        merges: dict[tuple[str, str], int] = {}
        target_merges = max(0, vocab_size - len(base_chars) - len(SPECIALS))
        stats = get_pair_stats()
        for step in range(target_merges):
            if not stats:
                break
            (a, b), freq = stats.most_common(1)[0]
            if freq < min_freq:
                break
            merges[(a, b)] = step
            merged = a + b
            # Apply this merge to every word and update stats incrementally.
            for w, syms in words.items():
                if a not in syms:
                    continue
                new_syms = []
                i = 0
                changed = False
                while i < len(syms):
                    if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                        new_syms.append(merged)
                        i += 2
                        changed = True
                    else:
                        new_syms.append(syms[i])
                        i += 1
                if changed:
                    words[w] = new_syms
            stats = get_pair_stats()
            if verbose and step % 500 == 0:
                print(f"  merge {step}/{target_merges}  '{a}'+'{b}' (freq {freq})")

        self.merges = merges

        # Build vocabulary: specials, base chars, then merged tokens in order.
        vocab_tokens = list(SPECIALS) + sorted(base_chars)
        for (a, b), _ in sorted(merges.items(), key=lambda kv: kv[1]):
            vocab_tokens.append(a + b)
        # de-dup preserving order
        seen = set()
        ordered = [t for t in vocab_tokens if not (t in seen or seen.add(t))]
        self.vocab = {t: i for i, t in enumerate(ordered)}
        self.inv_vocab = {i: t for t, i in self.vocab.items()}
        return self

    # ---- encoding / decoding -------------------------------------------
    def _encode_word(self, word: str) -> list[str]:
        if word.isalpha():
            syms = list(word) + [END]
        else:
            syms = [word]
        # greedily apply merges in learned-rank order
        while True:
            best_rank = None
            best_i = None
            for i in range(len(syms) - 1):
                r = self.merges.get((syms[i], syms[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            syms[best_i:best_i + 2] = [syms[best_i] + syms[best_i + 1]]
        return syms

    def encode(self, text: str) -> list[int]:
        ids = []
        for atom in _pre_tokenize(text):
            if atom == EOT:
                ids.append(self.vocab[EOT])
                continue
            for tok in self._encode_word(atom):
                ids.append(self.vocab.get(tok, self.vocab[UNK]))
        return ids

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            tok = self.inv_vocab.get(i, "")
            if tok == EOT:
                out.append(EOT)
            elif tok in (UNK,):
                out.append("")
            else:
                out.append(tok.replace(END, ""))
        return "".join(out)

    @property
    def size(self) -> int:
        return len(self.vocab)

    # ---- persistence ----------------------------------------------------
    def save(self, path: str) -> None:
        payload = {
            "vocab": self.vocab,
            "merges": [[a, b, r] for (a, b), r in self.merges.items()],
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path) as f:
            payload = json.load(f)
        t = cls()
        t.vocab = {k: int(v) for k, v in payload["vocab"].items()}
        t.inv_vocab = {v: k for k, v in t.vocab.items()}
        t.merges = {(a, b): int(r) for a, b, r in payload["merges"]}
        return t


if __name__ == "__main__":
    from .corpus import load_documents
    from .cutoff import apply_cutoff, build_training_text
    docs = load_documents()
    report = apply_cutoff(docs, 1900)
    text = build_training_text(report.admitted)
    tok = BPETokenizer().train(text[:2_000_000], vocab_size=2000, verbose=True)
    sample = "Whan that the whales did swim upon the sea,"
    ids = tok.encode(sample)
    print(f"\nvocab size {tok.size}")
    print("sample tokens:", [tok.inv_vocab[i] for i in ids][:20])
    print("roundtrip ok:", tok.decode(ids) == sample)
