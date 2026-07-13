# Phase 3A Model-Weight License Options

This is a practical engineering comparison, not legal advice. Corpus access, corpus redistribution, trained-weight copyright, database rights, share-alike effects, privacy, trademarks, and model-output obligations may differ by jurisdiction and require qualified review before a final decision.

## Apache-2.0

- **Redistribution and commercial use:** broad permission, including commercial use and modification, subject to license/notice conditions.
- **Derivative models:** generally easy to redistribute under compatible terms while preserving required notices.
- **Attribution:** retain copyright, license, and NOTICE material where applicable.
- **Patents:** includes an explicit contributor patent grant and termination provision; this is its main practical advantage over MIT.
- **Corpus obligations:** does not erase source-specific attribution or share-alike questions. A separate corpus-attribution package may still be required.
- **Hugging Face usability:** familiar and low-friction for tooling and downstream users.
- **Barriers:** requires clear ownership/contributor authority and confidence that permissive redistribution is intended.

## MIT

- **Redistribution and commercial use:** very broad and concise permission.
- **Derivative models:** low-friction; downstream users retain the notice.
- **Attribution:** copyright and permission notice must be included.
- **Patents:** no explicit patent grant, which can leave more uncertainty than Apache-2.0.
- **Corpus obligations:** source obligations remain separate and unresolved; MIT cannot be used to relabel corpus text.
- **Hugging Face usability:** excellent due to familiarity and simplicity.
- **Barriers:** minimal user friction, but that simplicity may be inadequate if patent or responsible-use terms are important.

## Responsible-AI or Open-Model License

- **Redistribution and commercial use:** depends on the specific license; use restrictions may limit some commercial or harmful applications.
- **Derivative models:** typically permitted only when downstream models preserve use restrictions or notices.
- **Attribution:** commonly required and may include model-card or acceptable-use obligations.
- **Patents:** varies; must be checked in the exact text.
- **Corpus obligations:** does not automatically solve source attribution, privacy, or share-alike analysis.
- **Hugging Face usability:** supported operationally, but restricted licenses may not qualify as open source and can complicate automated license filters.
- **Barriers:** compliance interpretation and downstream enforceability can deter users; the chosen restrictions must match a documented risk model.

## Custom Restrictive Research License

- **Redistribution and commercial use:** can prohibit commercial use or limit use to research.
- **Derivative models:** can require approval or prohibit redistribution.
- **Attribution:** fully customizable.
- **Patents:** custom language creates additional drafting and review burden.
- **Corpus obligations:** still cannot override upstream rights or cure unclear training-data permissions.
- **Hugging Face usability:** technically publishable with a custom license file, but creates the highest user and integration friction.
- **Barriers:** ambiguity, incompatibility, weak adoption, and significant legal maintenance. Use only if technically and legally necessary.

## Information Required Before Selection

1. Jurisdiction-specific analysis of whether and how trained weights are protected and who owns them.
2. Complete approved corpus registry, source licenses, attribution package, and analysis of CC BY-SA/GFDL implications.
3. Confirmation that all code and model contributors can grant the selected rights and any patent license.
4. Decision on commercial use, derivative models, hosted inference, fine-tuning, and redistribution goals.
5. Documented responsible-use requirements and whether contractual restrictions are necessary or enforceable.
6. Privacy, takedown, and provenance response plan.
7. Exact Hugging Face license metadata and model-card wording.
8. Qualified legal review of the final weight license and corpus obligations.

No model-weight license is finalized in Phase 3A.
