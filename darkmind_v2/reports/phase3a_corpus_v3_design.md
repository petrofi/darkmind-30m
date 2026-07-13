# Phase 3A Corpus v3 Design

Status: **DESIGN ONLY; NO CORPUS V3 DATA DOWNLOADED**

## Target Contract

The default target is 500,000,000 validated tokens, with decision checkpoints at 5M, 25M, 100M, 250M, and 500M tokens. The minimum experimental corpus is 100M tokens and the first serious base-training corpus is 250M tokens.

| Exclusive category | Target | Allowed range | 500M allocation |
|---|---:|---:|---:|
| High-quality Turkish prose | 60% | 55-65% | 300M |
| High-quality English prose | 25% | 20-30% | 125M |
| Technical and educational | 8% | 8-12% | 40M |
| Code and structured text | 4% | 3-7% | 20M |
| Controlled bilingual/dialogue | 3% | 0-5% | 15M |

Every accepted document is assigned to exactly one category. Language, source, and quality-tier fields remain orthogonal metadata; they do not create duplicate quota credit.

## Deterministic Pipeline

1. **Acquisition manifest:** require approved registry status, official HTTPS URL, pinned edition, expected bytes, license evidence, attribution policy, and checksum plan before download authorization.
2. **Immutable raw layer:** stream downloads to a temporary name, verify size and SHA-256, then publish an atomic completion marker. Raw inputs are never silently replaced.
3. **Strict decoding:** UTF-8 strict decoding, Unicode NFC normalization, replacement-character rejection, unsafe-control rejection, and mojibake detection.
4. **Structured extraction:** stream BZ2/ZIP/XML/TSV inputs, strip navigation/templates/boilerplate, preserve document boundaries, and retain source-specific provenance.
5. **Document quality:** language identification, line-length and character-distribution checks, repeated-character and excessive-punctuation filtering, low-information rejection, and code/prose classification.
6. **Privacy controls:** reject credentials, direct contact details, high-risk personal records, and private-conversation material; log only non-sensitive rejection metadata.
7. **Deduplication:** paragraph and document exact hashes first, then deterministic near-duplicate clustering. Cross-source dedup runs before quota allocation.
8. **Quota allocation:** enforce source caps, exclusive category targets, language targets, and quality tiers using a fixed seed and stable source/document ordering.
9. **Split isolation:** cluster document families before deterministic train/validation/eval assignment. No family or near-duplicate cluster may cross a split.
10. **Contamination control:** hash normalized evaluation prompts and close paraphrases; reject matching training spans and record the rejection reason.
11. **Tokenization:** use only frozen `darkmind_v2_sp_bpe24k_v1`; write deterministic uint16 shards with token/document boundary indexes.
12. **Final manifests:** SHA-256 every shard and manifest; record source, license, attribution, language, category, tier, rejection counts, and deterministic content hash.

## Quality Tiers

- **Tier A:** official and editorially controlled material with clear licensing and stable editions. Normal cleaning still applies.
- **Tier B:** open material accepted only after stronger boilerplate, duplication, quality, and privacy filtering.
- **Tier C:** experimental material such as short community-contributed bilingual sentences; tightly capped and separately audited.
- **Rejected:** unclear rights, private or social content, uncontrolled web crawl material, evaluation contamination, or failed quality/privacy gates.

## Source and Attribution Controls

Wikimedia text reuse must retain project/snapshot attribution and account for page-level imported material. Python documentation must retain PSF license notices and distinguish code examples. Tatoeba must retain sentence-level contributor and license metadata. Deferred sources do not become approved merely because their data is technically accessible.

The registry cites official evidence including Wikimedia's [Terms of Use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en#7._Licensing_of_Content), the [Python documentation license](https://docs.python.org/3.14/license.html), [Tatoeba terms](https://tatoeba.org/en/terms_of_use), and [MDN attribution guidance](https://developer.mozilla.org/en-US/docs/MDN/Writing_guidelines/Attrib_copyright_license).

## Storage and Loss Planning

- Likely raw-download budget for an approved 500M build: 35-60 GB, dominated by full Wikimedia dumps even when extraction stops at source caps.
- Processed normalized text plus per-document metadata: approximately 3-5 GiB.
- Raw uint16 token IDs at 500M: 1,000,000,000 bytes (0.931 GiB); shard indexes/manifests budget about 1.0 GiB total.
- Expected cleaning and cross-source deduplication loss: 20-35%; source estimates must not be treated as guaranteed yield.
- Keep at least 2x processed-storage headroom during atomic rebuilds and deterministic verification.

## Blocking Risks

The approved proposal is still short of Turkish prose, code, and technical diversity under the exclusive composition contract. Turkish institutional sources remain deferred because bulk model-training and redistribution terms are not established. Wikisource and Project Gutenberg need work-level rights checks. Stack Exchange and MDN need revision/file-level attribution and mixed-license handling. No production corpus build is approved until these gaps are resolved.
