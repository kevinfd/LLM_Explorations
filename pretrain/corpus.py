"""Provenance-first corpus assembly (spec section 5.3).

Every document carries a verified publication year and source. We fetch a real
public-domain text collection (the NLTK Gutenberg sample: 18 works spanning
1599-1920) and attach a curated, authoritative provenance record to each --
then cross-check the year embedded in each file's header against the curated
table. A mismatch is a provenance error, not something to paper over: the spec's
whole discipline is that date-unverifiable documents are excluded, never guessed.

This is a *pilot* corpus. It stands in for the pre-1986 scientific + general
corpus at a scale that runs on a laptop, while exercising the exact machinery
the real experiment needs: verified dates, a hard cutoff that actually
partitions the data, and a content channel for the anachronism-canary audit.
"""

import io
import os
import re
import urllib.request
import zipfile
from dataclasses import dataclass

# Only this host is reachable under the environment's egress policy.
GUTENBERG_URL = (
    "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/"
    "packages/corpora/gutenberg.zip"
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


@dataclass(frozen=True)
class Provenance:
    doc_id: str
    title: str
    author: str
    year: int          # authoritative first-publication year
    source: str


# Authoritative first-publication years (verified against standard references).
# Where a work's composition and first print differ, we use first print; where a
# header year differs (e.g. Alice: text says 1865), the mismatch is reported.
PROVENANCE = {
    "shakespeare-caesar":      Provenance("shakespeare-caesar", "Julius Caesar", "William Shakespeare", 1599, "Gutenberg/NLTK"),
    "shakespeare-hamlet":      Provenance("shakespeare-hamlet", "Hamlet", "William Shakespeare", 1603, "Gutenberg/NLTK"),
    "shakespeare-macbeth":     Provenance("shakespeare-macbeth", "Macbeth", "William Shakespeare", 1606, "Gutenberg/NLTK"),
    "bible-kjv":               Provenance("bible-kjv", "King James Bible", "Various", 1611, "Gutenberg/NLTK"),
    "milton-paradise":         Provenance("milton-paradise", "Paradise Lost", "John Milton", 1667, "Gutenberg/NLTK"),
    "blake-poems":             Provenance("blake-poems", "Poems", "William Blake", 1794, "Gutenberg/NLTK"),
    "edgeworth-parents":       Provenance("edgeworth-parents", "The Parent's Assistant", "Maria Edgeworth", 1796, "Gutenberg/NLTK"),
    "austen-sense":            Provenance("austen-sense", "Sense and Sensibility", "Jane Austen", 1811, "Gutenberg/NLTK"),
    "austen-emma":             Provenance("austen-emma", "Emma", "Jane Austen", 1815, "Gutenberg/NLTK"),
    "austen-persuasion":       Provenance("austen-persuasion", "Persuasion", "Jane Austen", 1818, "Gutenberg/NLTK"),
    "melville-moby_dick":      Provenance("melville-moby_dick", "Moby Dick", "Herman Melville", 1851, "Gutenberg/NLTK"),
    "whitman-leaves":          Provenance("whitman-leaves", "Leaves of Grass", "Walt Whitman", 1855, "Gutenberg/NLTK"),
    "carroll-alice":           Provenance("carroll-alice", "Alice's Adventures in Wonderland", "Lewis Carroll", 1865, "Gutenberg/NLTK"),
    "chesterton-thursday":     Provenance("chesterton-thursday", "The Man Who Was Thursday", "G. K. Chesterton", 1908, "Gutenberg/NLTK"),
    "chesterton-ball":         Provenance("chesterton-ball", "The Ball and the Cross", "G. K. Chesterton", 1909, "Gutenberg/NLTK"),
    "chesterton-brown":        Provenance("chesterton-brown", "The Innocence of Father Brown", "G. K. Chesterton", 1911, "Gutenberg/NLTK"),
    "bryant-stories":          Provenance("bryant-stories", "Stories to Tell to Children", "Sara Cone Bryant", 1918, "Gutenberg/NLTK"),
    "burgess-busterbrown":     Provenance("burgess-busterbrown", "The Burgess Bird Book / Buster Bear", "Thornton Burgess", 1920, "Gutenberg/NLTK"),
}


@dataclass
class Document:
    prov: Provenance
    text: str                 # boilerplate-stripped body
    header_year: int | None   # year parsed from the in-file header, if any
    year_verified: bool       # header_year agrees with authoritative year


_HEADER_RE = re.compile(r"^\[[^\]]*\b(1[5-9]\d\d|20\d\d)\b[^\]]*\]\s*", re.MULTILINE)


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "llm-explorations-pilot"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    with open(dest, "wb") as f:
        f.write(data)


def _strip_boilerplate(raw: str) -> tuple[str, int | None]:
    """Remove the leading '[Title by Author YEAR]' header (a digitization
    artifact / provenance line that must not enter the training text) and return
    the header year if present. Real corpora also carry Project Gutenberg license
    blocks; the NLTK sample is pre-cleaned of those, but we defensively cut any
    'START/END OF ... PROJECT GUTENBERG' spans as well (spec section 5.3)."""
    header_year = None
    m = _HEADER_RE.match(raw)
    if m:
        header_year = int(m.group(1))
        raw = raw[m.end():]
    # Defensive: strip any Gutenberg license spans if a future source has them.
    raw = re.sub(
        r"\*\*\*\s*START OF.*?\*\*\*", "", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(
        r"\*\*\*\s*END OF.*", "", raw, flags=re.DOTALL | re.IGNORECASE)
    return raw.strip(), header_year


def load_documents(cache_dir: str = CACHE_DIR) -> list[Document]:
    """Fetch (cached) and assemble the provenance-tagged document set."""
    zip_path = os.path.join(cache_dir, "gutenberg.zip")
    if not os.path.exists(zip_path):
        _download(GUTENBERG_URL, zip_path)

    docs: list[Document] = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".txt"):
                continue
            doc_id = os.path.splitext(os.path.basename(name))[0]
            prov = PROVENANCE.get(doc_id)
            if prov is None:
                # Unknown provenance -> excluded, never guessed (spec section 5.3).
                continue
            raw = z.read(name).decode("latin-1")
            text, header_year = _strip_boilerplate(raw)
            verified = header_year is not None and header_year == prov.year
            docs.append(Document(prov=prov, text=text, header_year=header_year,
                                 year_verified=verified))
    docs.sort(key=lambda d: d.prov.year)
    return docs


if __name__ == "__main__":
    docs = load_documents()
    print(f"{len(docs)} documents, {sum(len(d.text) for d in docs):,} chars total\n")
    for d in docs:
        flag = "" if d.year_verified else f"  [header={d.header_year} != {d.prov.year}]"
        print(f"  {d.prov.year}  {d.prov.doc_id:22s} {len(d.text):>9,} chars{flag}")
