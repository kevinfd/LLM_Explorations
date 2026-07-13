# Cutoff audit (cutoff = 1900)

Admitted 13 / 18 documents (10,274,085 chars).

## Metadata gate (authoritative publication year)

| status | year | doc | note |
|---|---|---|---|
| ADMIT | 1599 | shakespeare-caesar |  |
| ADMIT | 1603 | shakespeare-hamlet | header=1599 |
| ADMIT | 1606 | shakespeare-macbeth | header=1603 |
| ADMIT | 1611 | bible-kjv | header=None |
| ADMIT | 1667 | milton-paradise |  |
| ADMIT | 1794 | blake-poems | header=1789 |
| ADMIT | 1796 | edgeworth-parents | header=None |
| ADMIT | 1811 | austen-sense |  |
| ADMIT | 1815 | austen-emma | header=1816 |
| ADMIT | 1818 | austen-persuasion |  |
| ADMIT | 1851 | melville-moby_dick |  |
| ADMIT | 1855 | whitman-leaves |  |
| ADMIT | 1865 | carroll-alice |  |
| EXCLUDE | 1908 | chesterton-thursday | after cutoff |
| EXCLUDE | 1909 | chesterton-ball | after cutoff |
| EXCLUDE | 1911 | chesterton-brown | after cutoff |
| EXCLUDE | 1918 | bryant-stories | after cutoff |
| EXCLUDE | 1920 | burgess-busterbrown | after cutoff |

## Content gate (anachronism canaries)

No post-cutoff canary survived the metadata gate (8 canaries checked). Corpus certified period-pure.

## Provenance notes (header vs authoritative year)

- shakespeare-hamlet: file header year 1599 != authoritative 1603 (filtering uses authoritative year)
- shakespeare-macbeth: file header year 1603 != authoritative 1606 (filtering uses authoritative year)
- bible-kjv: file header year None != authoritative 1611 (filtering uses authoritative year)
- blake-poems: file header year 1789 != authoritative 1794 (filtering uses authoritative year)
- edgeworth-parents: file header year None != authoritative 1796 (filtering uses authoritative year)
- austen-emma: file header year 1816 != authoritative 1815 (filtering uses authoritative year)

**Corpus clean: True**