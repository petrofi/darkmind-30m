# DarkMind v2 Phase 1B Source Gap Decision

Status: source gap not resolved for tokenizer training yet.

This report evaluates the Phase 1B inventory gap without downloading new data, building the final corpus, training tokenizers, starting model training, or generating teacher data.

## Current Inventory

Original tokenizer pilot target:

- Total normalized characters: 50,000,000
- Turkish target: 30,000,000
- English target: 20,000,000
- Single-source cap: 40%, or 20,000,000 characters at the 50M target

Completed usable normalized characters:

| Source | Turkish | English | Total |
| --- | ---: | ---: | ---: |
| wikimedia_trwiki_20260601_articles_p1p1500000 | 19,983,785 | 0 | 19,983,785 |
| python_docs_tr_3_14_6_text | 2,281,545 | 0 | 2,281,545 |
| python_docs_en_3_14_6_text | 0 | 7,480,086 | 7,480,086 |
| tatoeba_sentences_detailed_20260704 | 6,752,531 | 8,227,532 | 14,980,063 |
| **Total** | **29,017,861** | **15,707,618** | **44,725,479** |

Current gaps:

- Total gap: 5,274,521 characters
- Turkish gap: 982,139 characters
- English gap: 4,292,382 characters

The 50M target is not feasible yet because the approved downloaded source set is short overall and cannot satisfy the 60/40 language split. The English shortage is the main blocker. Turkish is close, but still short enough that silently lowering the target or language ratio would weaken the gate.

Tokenizer training should not start yet because the target is explicitly failing, the English shortfall is material, and the current corpus mix would overuse Tatoeba-style short sentences and Python documentation relative to the intended balanced pilot.

## Source-Cap And Content-Mix Risks

No current source exceeds the 20M cap at the 50M target:

- trwiki: 19,983,785, very close to the 20M cap
- Tatoeba: 14,980,063
- English Python docs: 7,480,086
- Turkish Python docs: 2,281,545

The cap risk is concentrated in trwiki because it has only about 16k characters of headroom at the 50M cap. Adding more Turkish Wikipedia split data would not be safe as a source-gap fix because it would effectively bypass the intended single-source limit.

Content-mix risks:

- Tatoeba currently contributes 14.98M characters, much higher than the original short-sentence target ratio.
- Python docs provide useful technical coverage but cannot carry the English shortage alone.
- Turkish technical documentation is far smaller than planned.
- Additional sources should preferably add official, license-clear English educational/general prose and small Turkish non-Wikipedia Wikimedia prose to avoid overusing trwiki or Tatoeba.

## Path A - Keep 50M Target

Path A keeps the original 50M target, keeps the 60/40 language target, and keeps the 40% source cap. It requires additional official, license-clear sources before any tokenizer training.

Official license basis for Wikimedia candidates:

- Wikimedia Terms of Use require text contributions to be under CC BY-SA 4.0 and GFDL unless a project edition requires a different free license.
- Wikimedia Terms state that compliant commercial reuse is allowed.
- Project footers for the Turkish Wikibooks and Turkish Wikivoyage pages checked during review show Creative Commons BY-SA 4.0 footer text.
- Official dump URLs were checked with HEAD requests only; no corpus data was downloaded.

### Candidate A1 - English Wikiversity

- source_id: `wikimedia_enwikiversity_20260601_articles`
- source_name: English Wikiversity pages-articles official dump
- official homepage: `https://en.wikiversity.org/`
- official download URL: `https://dumps.wikimedia.org/enwikiversity/20260601/enwikiversity-20260601-pages-articles.xml.bz2`
- language: `en`
- content type: educational articles
- license ID: `CC-BY-SA-4.0-and-GFDL`
- official license URL: `https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use`
- attribution requirements: preserve page ID, revision ID, title, source ID, license ID, and page URL
- redistribution implications: share-alike and attribution obligations must be preserved
- commercial-use status: allowed with license compliance
- modification status: allowed with license compliance
- source version or snapshot date: `enwikiversity 20260601 pages-articles`, snapshot `2026-06-01`
- estimated download size: 114,256,832 compressed bytes
- estimated usable normalized characters: 5,000,000
- personal-data risk: low-to-medium; open collaborative educational pages, with URL/PII filters still required
- encoding risk: medium; Wikimedia wikitext must pass mojibake/replacement-character gates
- duplication risk: medium; potential overlap with Wikipedia-style explanations and Python docs
- decision: approved and added to registry
- rationale: small enough for the existing download budget and likely enough to close the English gap without relying on Simple English Wikipedia's much larger dump

### Candidate A2 - Turkish Wikibooks

- source_id: `wikimedia_trwikibooks_20260601_articles`
- source_name: Turkish Wikibooks pages-articles official dump
- official homepage: `https://tr.wikibooks.org/`
- official download URL: `https://dumps.wikimedia.org/trwikibooks/20260601/trwikibooks-20260601-pages-articles.xml.bz2`
- language: `tr`
- content type: instructional books
- license ID: `CC-BY-SA-4.0`
- official license URL: `https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use`
- attribution requirements: preserve page ID, revision ID, title, source ID, license ID, and page URL
- redistribution implications: share-alike and attribution obligations must be preserved
- commercial-use status: allowed with license compliance
- modification status: allowed with license compliance
- source version or snapshot date: `trwikibooks 20260601 pages-articles`, snapshot `2026-06-01`
- estimated download size: 3,030,087 compressed bytes
- estimated usable normalized characters: 1,000,000
- personal-data risk: low-to-medium; open collaborative pages, with URL/PII filters still required
- encoding risk: medium; Turkish wikitext must pass strict Unicode and mojibake gates
- duplication risk: medium; may overlap with technical/instructional docs
- decision: approved and added to registry
- rationale: small official Turkish source that can cover most or all of the Turkish shortage without adding more Turkish Wikipedia

### Candidate A3 - Turkish Wikivoyage

- source_id: `wikimedia_trwikivoyage_20260601_articles`
- source_name: Turkish Wikivoyage pages-articles official dump
- official homepage: `https://tr.wikivoyage.org/`
- official download URL: `https://dumps.wikimedia.org/trwikivoyage/20260601/trwikivoyage-20260601-pages-articles.xml.bz2`
- language: `tr`
- content type: travel guide articles
- license ID: `CC-BY-SA-4.0`
- official license URL: `https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use`
- attribution requirements: preserve page ID, revision ID, title, source ID, license ID, and page URL
- redistribution implications: share-alike and attribution obligations must be preserved
- commercial-use status: allowed with license compliance
- modification status: allowed with license compliance
- source version or snapshot date: `trwikivoyage 20260601 pages-articles`, snapshot `2026-06-01`
- estimated download size: 1,429,990 compressed bytes
- estimated usable normalized characters: 700,000
- personal-data risk: low-to-medium; travel listings can contain external links or contact-like strings, so URL/PII filters are required
- encoding risk: medium; Turkish wikitext must pass strict Unicode and mojibake gates
- duplication risk: low-to-medium; mostly travel prose, but location boilerplate must be filtered
- decision: approved and added to registry
- rationale: small official Turkish prose source that provides headroom if Turkish Wikibooks does not produce the full 1M usable characters

### Candidate A4 - Simple English Wikipedia

- source_id: `wikimedia_simplewiki_20260601_articles`
- source_name: Simple English Wikipedia pages-articles official dump
- official homepage: `https://simple.wikipedia.org/`
- official download URL: `https://dumps.wikimedia.org/simplewiki/20260601/simplewiki-20260601-pages-articles.xml.bz2`
- language: `en`
- content type: encyclopedic articles
- license ID: `CC-BY-SA-4.0-and-GFDL`
- official license URL: `https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use`
- attribution requirements: preserve page ID, revision ID, title, source ID, license ID, and page URL
- redistribution implications: share-alike and attribution obligations must be preserved
- commercial-use status: allowed with license compliance
- modification status: allowed with license compliance
- source version or snapshot date: `simplewiki 20260601 pages-articles`, snapshot `2026-06-01`
- estimated download size: 349,657,203 compressed bytes
- estimated usable normalized characters: likely sufficient for the English gap, but not inventoried
- personal-data risk: low-to-medium; open collaborative pages, with URL/PII filters still required
- encoding risk: medium; Wikimedia wikitext must pass strict Unicode and mojibake gates
- duplication risk: medium-to-high with existing Wikipedia-style content
- decision: deferred
- rationale: license and official-source posture are good, but the compressed dump is large enough to create avoidable download-budget pressure. English Wikiversity is a smaller first supplemental source.

Path A projected capacity after the approved additions:

- Existing Turkish: 29,017,861
- Added Turkish cap estimate: 1,700,000
- Projected Turkish capacity: 30,717,861
- Existing English: 15,707,618
- Added English cap estimate: 5,000,000
- Projected English capacity: 20,707,618
- Projected usable capacity: 51,425,479

This is sufficient to keep the 50M target if the next inventory confirms the estimates. The next inventory must still enforce exact source caps, language targets, mojibake rejection, attribution metadata, and duplicate gates.

## Path B - Explicit Reduced Approved-Only Pilot

Path B would create a smaller approved-only pilot using only already downloaded sources. This would avoid new downloads, but the evaluated targets do not satisfy the current 60/40 language rule.

| Candidate target | TR target 60% | EN target 40% | TR available | EN available | Feasible under 60/40? | Source-cap notes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 40M | 24,000,000 | 16,000,000 | 29,017,861 | 15,707,618 | No, English short by 292,382 | Cap is 16M; trwiki would need sampling below its available 19.98M |
| 42M | 25,200,000 | 16,800,000 | 29,017,861 | 15,707,618 | No, English short by 1,092,382 | Cap is 16.8M; trwiki would need sampling below its available 19.98M |
| 44M | 26,400,000 | 17,600,000 | 29,017,861 | 15,707,618 | No, English short by 1,892,382 | Cap is 17.6M; trwiki would need sampling below its available 19.98M |

Path B risks:

- A 40M/42M/44M target would still fail strict 60/40 because English is short.
- Relaxing the English ratio would silently weaken the language target, which is not allowed.
- Tatoeba would remain highly represented relative to the original short-sentence mix.
- Python documentation would remain a large fraction of the English technical slice.
- Any reduced target should be labeled a "pilot tokenizer experiment", not a final base tokenizer corpus.
- A reduced pilot could still be useful later, but only after explicit user approval of the target and the language-mix exception.

Path B decision: not recommended for this step.

## Recommendation

Recommendation: A. Add approved supplemental sources and keep the 50M target.

Why this is safer:

- It preserves the original 50M target.
- It preserves the 60/40 language target.
- It preserves the 40% single-source cap.
- It avoids using additional Turkish Wikipedia split data to work around the trwiki cap.
- It uses official, versioned, license-clear Wikimedia dumps with manageable compressed sizes.
- It avoids silently labeling an underfilled or language-skewed corpus as successful.

Remaining risk:

- Estimated usable characters may be lower than projected after filtering.
- Wikiversity/Wikibooks/Wikivoyage content may add educational/travel/book style bias.
- Wikimedia-family content will be larger overall, so final source/content-mix reporting must make that explicit.
- The next inventory may still fail, in which case Phase 1B should stop again before tokenizer training.

Can tokenizer training start after this step?

No. Tokenizer training still must not start. The next safe step is to download only the newly approved supplemental sources, then rerun source-specific inventory and inspect the updated report. Tokenizer training can start only after the 50M/60-40/source-cap gates pass and the final corpus build is explicitly approved.

Exact next command after user approval to download the new sources:

```powershell
Set-Location 'C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-v2-phase1b'
python darkmind_v2/corpus/download_phase1b_sources.py `
  --source-id wikimedia_enwikiversity_20260601_articles `
  --source-id wikimedia_trwikibooks_20260601_articles `
  --source-id wikimedia_trwikivoyage_20260601_articles
```

Do not run that command until the user explicitly approves downloading the new sources.
