"""Hard cutoff filter + anachronism-canary audit (spec sections 5.3, 8.4).

Two independent gates, matching the spec's "certify the negative" requirement:

  1. METADATA GATE (provenance): a document is admitted iff its *authoritative*
     publication year <= cutoff. This is the primary filter.

  2. CONTENT GATE (canary audit): scan the admitted corpus for tokens that could
     only appear after the cutoff -- named entities and coinages from the
     post-cutoff works. If any survive, the metadata gate leaked and the run is
     invalid. This is the analog of the spec's canary sweep for post-1986
     coinages ("backpropagation", "perestroika", ...); here the canaries are
     names that appear only in the 1908-1920 works.

The audit is deliberately conservative: it reports every canary hit with its
document, so a reviewer can see exactly what (if anything) leaked.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

from .corpus import Document


# Canaries: surface forms that occur ONLY in post-1900 works in this corpus.
# In the real experiment these are post-1986 coinages/events; here they are
# character and title tokens unique to the Chesterton/Bryant/Burgess works.
POST_CUTOFF_CANARIES = [
    "Buster", "Chesterton", "Syme",          # Chesterton's The Man Who Was Thursday
    "MacIan", "Turnbull",                     # The Ball and the Cross
    "Flambeau", "Valentin",                   # Father Brown
    "Gregory",                                # Thursday antagonist
]


def _word_set(text: str) -> Counter:
    return Counter(re.findall(r"[A-Za-z][A-Za-z'-]*", text))


@dataclass
class FilterReport:
    cutoff_year: int
    admitted: list[Document]
    excluded: list[Document]
    provenance_notes: list[str] = field(default_factory=list)
    canary_hits: dict[str, list[str]] = field(default_factory=dict)  # canary -> [doc_id...]
    clean: bool = False

    def render(self) -> str:
        lines = [f"# Cutoff audit (cutoff = {self.cutoff_year})", ""]
        lines.append(f"Admitted {len(self.admitted)} / "
                     f"{len(self.admitted) + len(self.excluded)} documents "
                     f"({sum(len(d.text) for d in self.admitted):,} chars).\n")
        lines.append("## Metadata gate (authoritative publication year)\n")
        lines.append("| status | year | doc | note |")
        lines.append("|---|---|---|---|")
        for d in self.admitted:
            note = "" if d.year_verified else f"header={d.header_year}"
            lines.append(f"| ADMIT | {d.prov.year} | {d.prov.doc_id} | {note} |")
        for d in self.excluded:
            lines.append(f"| EXCLUDE | {d.prov.year} | {d.prov.doc_id} | after cutoff |")
        lines.append("\n## Content gate (anachronism canaries)\n")
        if self.canary_hits:
            lines.append("LEAK DETECTED -- post-cutoff tokens found in admitted corpus:\n")
            for canary, docs in sorted(self.canary_hits.items()):
                lines.append(f"- `{canary}` in {', '.join(docs)}")
        else:
            lines.append(f"No post-cutoff canary survived the metadata gate "
                         f"({len(POST_CUTOFF_CANARIES)} canaries checked). Corpus certified period-pure.")
        if self.provenance_notes:
            lines.append("\n## Provenance notes (header vs authoritative year)\n")
            lines.extend(f"- {n}" for n in self.provenance_notes)
        lines.append(f"\n**Corpus clean: {self.clean}**")
        return "\n".join(lines)


def apply_cutoff(docs: list[Document], cutoff_year: int,
                 canaries: list[str] = POST_CUTOFF_CANARIES) -> FilterReport:
    admitted = [d for d in docs if d.prov.year <= cutoff_year]
    excluded = [d for d in docs if d.prov.year > cutoff_year]

    notes = [
        f"{d.prov.doc_id}: file header year {d.header_year} != authoritative "
        f"{d.prov.year} (filtering uses authoritative year)"
        for d in admitted if not d.year_verified
    ]

    # Content gate: whole-word canary scan over the admitted corpus only.
    hits: dict[str, list[str]] = {}
    admitted_words = {d.prov.doc_id: _word_set(d.text) for d in admitted}
    for canary in canaries:
        for doc_id, words in admitted_words.items():
            if words.get(canary, 0) > 0:
                hits.setdefault(canary, []).append(doc_id)

    report = FilterReport(
        cutoff_year=cutoff_year, admitted=admitted, excluded=excluded,
        provenance_notes=notes, canary_hits=hits, clean=(len(hits) == 0),
    )
    return report


def build_training_text(admitted: list[Document]) -> str:
    """Concatenate admitted documents with a document separator token."""
    return "\n\n<|endoftext|>\n\n".join(d.text for d in admitted)


if __name__ == "__main__":
    from .corpus import load_documents
    docs = load_documents()
    report = apply_cutoff(docs, cutoff_year=1900)
    print(report.render())
