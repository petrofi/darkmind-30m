# DarkMind v2 Phase 1B Source Inventory After Supplemental

Result: **PASS**

**FINAL CORPUS BUILD IS READY FOR USER APPROVAL**

## Target Feasibility

- Total usable normalized characters: 51326413
- Turkish usable characters: 30638353
- English usable characters: 20688060
- 50M total target feasible: True
- Turkish 30M target feasible: True
- English 20M target feasible: True
- Any source exceeds 40% / 20M cap: False
- License and attribution metadata complete: True
- Final corpus build can start next after user approval: True

The inventory now has more than the target capacity. The final corpus builder still must sample down to the exact approved 50M target, 30M Turkish target, 20M English target, and per-source caps before tokenizer training.

## Download Summary

- Selected supplemental downloaded bytes: 118716909
- Total downloaded bytes after supplemental downloads: 783549056
- Remaining bytes under 1GB cap: 216450944
- Hard download cap bytes: 1000000000

| Supplemental source | Bytes | Official checksum verified | SHA-256 |
| --- | ---: | --- | --- |
| wikimedia_enwikiversity_20260601_articles | 114256832 | True | 6f375b4a2dbbf171b3a996b1738583c279760fc74b3756bf1ba62d26d2f5d0ee |
| wikimedia_trwikibooks_20260601_articles | 3030087 | True | 35fa367a34e963cbeca5e18cce9903507fce86903f101aadad14a8f5dcccbf38 |
| wikimedia_trwikivoyage_20260601_articles | 1429990 | True | 3e76253a7fc37a51b675ea8e0dc33ab4fa1e4437aab18250c7e72dd9237322c3 |

## Usable Character Inventory

| Source | Usable chars | TR chars | EN chars | Accepted docs | Rejected docs | License metadata | Attribution metadata | Source cap OK |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| python_docs_en_3_14_6_text | 7480086 | 0 | 7480086 | 58197 | 8394 | 66591/66591 | 66591/66591 | True |
| python_docs_tr_3_14_6_text | 2281545 | 2281545 | 0 | 17202 | 77450 | 94652/94652 | 94652/94652 | True |
| tatoeba_sentences_detailed_20260704 | 14980063 | 6752531 | 8227532 | 263653 | 589180 | 852833/852833 | 852833/852833 | True |
| wikimedia_enwikiversity_20260601_articles | 4980442 | 0 | 4980442 | 6834 | 1157 | 7991/7991 | 7991/7991 | True |
| wikimedia_trwiki_20260601_articles_p1p1500000 | 19983785 | 19983785 | 0 | 20290 | 3108 | 23398/23398 | 23398/23398 | True |
| wikimedia_trwikibooks_20260601_articles | 980610 | 980610 | 0 | 610 | 111 | 721/721 | 721/721 | True |
| wikimedia_trwikivoyage_20260601_articles | 639882 | 639882 | 0 | 515 | 49 | 564/564 | 564/564 | True |

## Supplemental Results

- `wikimedia_enwikiversity_20260601_articles`: 4,980,442 usable English characters.
- `wikimedia_trwikibooks_20260601_articles`: 980,610 usable Turkish characters.
- `wikimedia_trwikivoyage_20260601_articles`: 639,882 usable Turkish characters.

## Stop Point

No final corpus was built. No tokenizer was trained. No tokenizer candidates were compared. No model training or Qwen generation was started.

