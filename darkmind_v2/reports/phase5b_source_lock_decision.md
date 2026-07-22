# Phase 5B source-lock decision

## Decision

**PARTIALLY LOCKED**

The registry is materially stronger: all 20 candidates now have official-source evidence records, explicit acceptance gates, conservative capacities, risks and resolution steps. Three reproducible sources are approved and their acquisition entries have exact IDs, URLs, versions, filenames, size ranges, checksums, licenses, attribution, rate/retry policies and destination templates. The approved acquisition manifest validates, and concentration limits pass.

The plan is not ready for acquisition:

- Approved expected pre-dedup capacity is **10.0M**, below the **250M** lock threshold by **240.0M**.
- Approved conservative post-filter capacity is **6.3M**, below the **200M** lock threshold by **193.7M**.
- Turkish general, English general and controlled bilingual have zero approved capacity.
- Technical documentation has 3.8M conservative capacity against 42M.
- Code/structured text has 2.5M conservative capacity against 26M.
- DGT-Acquis does not provide verified Turkish coverage.
- Major conditional sources still lack exact artifact manifests, collection rights, attribution implementations, sample-based yield estimates or overlap evidence.

## Exact remaining actions

1. Resolve GOV.UK Content API, Turk Kutuphaneciligi, NIST, EUR-Lex, DergiPark, PMC, OpenStax, Stack Exchange, GitHub and arXiv steps recorded in the registry.
2. Identify additional official, explicitly reusable Turkish general/educational and English general sources; current conditional lower bounds cannot fill those categories.
3. Replace DGT-Acquis as the bilingual candidate with a reproducible Turkish-English artifact.
4. Prefer low-PII official code/tutorial archives over relying on Stack Exchange or broad GitHub volume.
5. Perform metadata-only inventories first, then obtain explicit authorization for small source-specific samples to measure extraction and rejection rates.
6. Promote a source only after snapshot, license, acquisition, attribution, checksum, quality, PII and overlap gates all pass.
7. Recompute category and concentration tables. A lock requires at least 250M approved expected and 200M approved conservative tokens with every category covered.
8. Bind $EXTERNAL_SSD_ROOT to a verified volume and approve execution in a new manifest revision only after the source plan reaches LOCKED.

No download is permitted under this decision. The continuation pilot remains planning-only and requires a future LOCKED source plan.

**DARKMIND V2 CORPUS V4 SOURCE PLAN REQUIRES ADDITIONAL LICENSE OR CAPACITY RESOLUTION**
