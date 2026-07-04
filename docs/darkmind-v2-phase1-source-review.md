# DarkMind v2 Phase 1A Source Review

Phase 1A is a planning and validation phase only. No corpus was downloaded, no tokenizer was trained, and no model training was started.

The goal is to define which sources may be considered for a small Turkish-English tokenizer pilot only after official source, license, attribution, redistribution, commercial-use, modification, size, and checksum evidence has been recorded in `darkmind_v2/corpus/source_registry.example.json`.

## Approval Rules

A source can be considered for Phase 1B only if it has:

- an official homepage,
- an official download URL or documented official retrieval method,
- official license evidence,
- explicit attribution and redistribution requirements,
- commercial-use and modification status,
- a pinned source version or snapshot date,
- an estimated download size,
- a hard retrieval cap,
- an intended sample cap,
- an approval reason,
- a risk level.

Common Crawl-derived datasets are not automatically approved. Social, private, leaked, personal, Reddit, chat-log, and profile-derived datasets are rejected for this phase.

## Recommended For The Tokenizer Pilot

| Candidate | Language | Role | Status | Evidence |
| --- | --- | --- | --- | --- |
| Turkish Wikipedia pages-articles official dump split | tr | Primary Turkish general prose | Recommended with medium risk | [trwiki dumps](https://dumps.wikimedia.org/trwiki/latest/), [Wikimedia Terms of Use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use) |
| Python documentation English | en | Technical prose and code-adjacent text | Recommended with low risk | [Python docs](https://docs.python.org/3/), [Python documentation license](https://docs.python.org/3/license.html) |
| Python documentation Turkish | tr | Turkish technical prose | Recommended with low risk | [Turkish Python docs](https://docs.python.org/tr/3/), [Python documentation license](https://docs.python.org/3/license.html) |
| Tatoeba official sentence exports | mixed_tr_en | Short sentence supplement | Recommended with medium risk | [Tatoeba downloads](https://tatoeba.org/en/downloads), [Tatoeba terms](https://tatoeba.org/en/terms_of_use) |

### Turkish Wikipedia

Turkish Wikipedia is useful because it provides broad Turkish prose from an official Wikimedia dump endpoint. The Phase 1A plan uses one pages-articles split chunk instead of the full dump so the retrieval remains below the 1GB hard cap. The registry records Wikimedia licensing as CC BY-SA 4.0 and GFDL terms and requires attribution/share-alike handling in manifests and reports.

Risks:

- wiki markup and template noise,
- article duplication and boilerplate,
- license attribution complexity,
- overrepresentation if it exceeds the 40% source cap.

Controls:

- use only official dump URLs,
- verify checksums before any future use,
- remove non-article namespaces, templates, tables, and navigation,
- cap the source at 20M normalized characters.

### Python Documentation

The official Python documentation is useful for technical prose, API wording, and small code-adjacent examples. The English and Turkish documentation pages are kept as separate source entries because they serve different language-balancing roles.

Risks:

- duplicated navigation and generated index pages,
- version drift if pages are not pinned,
- separate license treatment for documentation and examples.

Controls:

- pin exact Python documentation version before retrieval,
- preserve PSF documentation license and example-code license notes,
- strip navigation chrome and duplicated API index content,
- keep code examples as a small fraction of the tokenizer pilot.

### Tatoeba

Tatoeba is useful as a small sentence-style supplement, not as the backbone of the corpus. It can improve short Turkish-English sentence coverage if sentence IDs and attribution are preserved.

Risks:

- user-contributed text can include duplicates and uneven style,
- attribution must be preserved,
- sentence-level data should not dominate tokenizer training.

Controls:

- use official exports only,
- filter to Turkish and English text fields,
- keep only sentence text and IDs,
- cap the sample at 3M normalized characters.

## Deferred Sources

| Candidate | Reason Deferred | Required Before Use |
| --- | --- | --- |
| English Wikipedia dump chunks | Full English dumps are far above the 1GB pilot cap; a chunked sample could be useful later. | Pin exact chunk URLs, checksum files, sample limits, and attribution plan. |
| Project Gutenberg public-domain texts | Many works are public domain in the United States, but jurisdiction and Project Gutenberg trademark/header handling need exact per-work review. | Select specific works, record official license evidence, strip Project Gutenberg boilerplate only according to its terms, and document jurisdiction assumptions. |
| Wikisource Turkish/English | Potentially useful prose, but it needs per-source license and quality review. | Verify official dump/source, license, metadata, and deduplication behavior. |
| Stack Exchange data dump | CC BY-SA licensing can be usable, but attribution and PII/user-content risks are higher. | Separate legal/attribution review and PII filtering plan. Not part of Phase 1A pilot. |

## Rejected For Phase 1A

| Source Type | Decision | Reason |
| --- | --- | --- |
| Common Crawl-derived datasets such as FineWeb, OSCAR, CulturaX, mC4, CC100, or RefinedWeb | Rejected for automatic approval | Source-level licensing and content provenance are too broad for this pilot without a separate review. |
| Scraped news, blogs, forums, social media, chat logs, or comments | Rejected | Licensing, privacy, and redistribution evidence is not reliably source-specific. |
| Leaked, private, personal, or profile-derived datasets | Rejected | Not compatible with the project safety and data policy. |
| Reddit-derived data | Rejected | The project policy forbids Reddit training data. |
| Unofficial mirrors of official documentation | Rejected | Official source and license evidence must come from the original host. |

## Phase 1B Recommendation

Phase 1B should begin with the approved registry entries only, then stop after manifest generation and validation. The first pilot corpus should target 50M normalized characters with a 60/40 Turkish-English split, source cap of 40%, and a hard 1GB download cap. If English general prose remains underfilled, approve a small official English Wikipedia chunk or a carefully reviewed public-domain text subset before tokenizer training is considered.
