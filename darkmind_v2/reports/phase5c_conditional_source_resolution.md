# Phase 5C Conditional Source Resolution

Planning-only technical due diligence; not legal advice. No data was downloaded.

Resolved Phase 5B conditional records: **11/11**.

## govuk_content_ogl3_20260722

Final status: **CONDITIONAL**
Resolution path: `metadata_resolvable`
Reason: Pin a Content API URL inventory and exclude third-party or personal-data pages.

| # | Required question | Recorded answer |
|---:|---|---|
| 1 | `exact_downloadable_artifact` | Content API endpoint inventory not yet materialized |
| 2 | `exact_snapshot_release_dump_tag_or_commit` | unresolved_Content_API_URL_inventory_and_response_hashes |
| 3 | `content_license_explicit` | OGL v3 with listed personal-data, logo, identity, patent, and third-party exclusions |
| 4 | `collection_or_database_license_explicit` | not_applicable_to_page_content |
| 5 | `redistribution_rights_explicit` | commercial reuse permitted for covered public-sector information |
| 6 | `modification_and_derivative_processing_rights_explicit` | adaptation permitted with source acknowledgement and no misleading status |
| 7 | `machine_processing_or_model_training_prohibited` | none identified in OGL |
| 8 | `attribution_obligations_executable` | OGL attribution plus provider-specific statements |
| 9 | `official_acquisition_reproducible` | official GOV.UK Content API; no HTML scraping |
| 10 | `checksum_or_signed_manifest_available` | must generate immutable URL, ETag, last-modified, byte-size, and SHA-256 manifest before approval |
| 11 | `language_composition_officially_supported` | {"en": 1.0} |
| 12 | `corpus_v3_relationship_understood` | new English institutional prose |
| 13 | `conservative_capacity_estimable_without_download` | Phase 5A estimate reduced for rights exclusions, templates, live-page drift, language and cross-page deduplication |
| 14 | `source_concentration_violation` | False |
| 15 | `benchmark_pii_private_or_contamination_concerns` | {"benchmark_contamination": "low", "boilerplate": "high", "copyrighted_long_form": "medium", "pii": "medium", "private_data": "low", "spam": "low", "unsafe_content": "low"} |

Next action: Pin a Content API URL inventory and exclude third-party or personal-data pages.

## turk_kutuphaneciligi_ccby_20260722

Final status: **CONDITIONAL**
Resolution path: `bounded_sample_required`
Reason: The exact issue/article inventory and item-level CC BY evidence remain unresolved.

| # | Required question | Recorded answer |
|---:|---|---|
| 1 | `exact_downloadable_artifact` | archive visible; immutable article inventory not recorded |
| 2 | `exact_snapshot_release_dump_tag_or_commit` | unresolved dated OAI/article ID inventory |
| 3 | `content_license_explicit` | journal policy states CC BY 4.0 |
| 4 | `collection_or_database_license_explicit` | DergiPark collection/database terms not established for bulk reuse |
| 5 | `redistribution_rights_explicit` | article text may be reused where CC BY 4.0 is explicit |
| 6 | `modification_and_derivative_processing_rights_explicit` | CC BY 4.0 adaptation permitted with attribution and change notice |
| 7 | `machine_processing_or_model_training_prohibited` | none identified in CC BY policy |
| 8 | `attribution_obligations_executable` | author, title, journal, DOI/URL, license and modification notice |
| 9 | `official_acquisition_reproducible` | official journal archive/OAI only after permission and rate-limit confirmation |
| 10 | `checksum_or_signed_manifest_available` | article manifest and generated SHA-256 required |
| 11 | `language_composition_officially_supported` | {"en": 0.15, "tr": 0.85} |
| 12 | `corpus_v3_relationship_understood` | new non-Wikimedia Turkish scholarly prose |
| 13 | `conservative_capacity_estimable_without_download` | long-running bilingual archive estimate discounted for article-license uncertainty, references, metadata PII and deduplication |
| 14 | `source_concentration_violation` | False |
| 15 | `benchmark_pii_private_or_contamination_concerns` | {"benchmark_contamination": "medium", "boilerplate": "medium", "copyrighted_long_form": "medium", "pii": "medium", "private_data": "low", "spam": "low", "unsafe_content": "low"} |

Next action: The exact issue/article inventory and item-level CC BY evidence remain unresolved.

Future bounded sample (not authorized):
maximum 25,000,000 bytes; destination `$EXTERNAL_SSD_ROOT\DarkMindArchive\corpus-v4\samples\turk_kutuphaneciligi_ccby_20260722`; SHA-256 required; training use is false.

## nist_publications_cleared_subset

Final status: **CONDITIONAL**
Resolution path: `metadata_resolvable`
Reason: Pin a publication inventory and exclude every marked third-party work.

| # | Required question | Recorded answer |
|---:|---|---|
| 1 | `exact_downloadable_artifact` | not yet limited to a reproducible employee-authored subset |
| 2 | `exact_snapshot_release_dump_tag_or_commit` | unresolved publication identifier allowlist |
| 3 | `content_license_explicit` | NIST employee works generally public domain in the United States; Technical Series has stated reuse/citation terms |
| 4 | `collection_or_database_license_explicit` | unknown_pending_collection_inventory_terms |
| 5 | `redistribution_rights_explicit` | available only after author and third-party-rights filtering |
| 6 | `modification_and_derivative_processing_rights_explicit` | varies by item; third-party material excluded |
| 7 | `machine_processing_or_model_training_prohibited` | none identified for cleared works |
| 8 | `attribution_obligations_executable` | publication title, NIST authors, report identifier and source |
| 9 | `official_acquisition_reproducible` | official NIST publication pages/files only |
| 10 | `checksum_or_signed_manifest_available` | generated SHA-256 and item-rights record required |
| 11 | `language_composition_officially_supported` | {"en": 1.0} |
| 12 | `corpus_v3_relationship_understood` | new English scientific and engineering prose |
| 13 | `conservative_capacity_estimable_without_download` | broad official publication volume discounted for employee-authorship, third-party figures/text, PDF extraction and near deduplication |
| 14 | `source_concentration_violation` | False |
| 15 | `benchmark_pii_private_or_contamination_concerns` | {"benchmark_contamination": "medium", "boilerplate": "medium", "copyrighted_long_form": "medium", "pii": "low", "private_data": "low", "spam": "low", "unsafe_content": "low"} |

Next action: Pin a publication inventory and exclude every marked third-party work.

## eur_lex_reusable_text_20260722

Final status: **CONDITIONAL**
Resolution path: `metadata_resolvable`
Reason: Pin a CELLAR/data-dump release and preserve document-level reuse notices.

| # | Required question | Recorded answer |
|---:|---|---|
| 1 | `exact_downloadable_artifact` | CELEX sector 3 English in-force subset proposed but not pinned |
| 2 | `exact_snapshot_release_dump_tag_or_commit` | unresolved CELEX sector/language/dump date and checksum inventory |
| 3 | `content_license_explicit` | legal documents reusable unless special conditions apply; EU editorial material CC BY 4.0 |
| 4 | `collection_or_database_license_explicit` | EUR-Lex metadata CC0; collection contents retain document-specific rights |
| 5 | `redistribution_rights_explicit` | permitted for eligible EU-owned documents with source acknowledgement |
| 6 | `modification_and_derivative_processing_rights_explicit` | permitted for eligible content with modifications indicated |
| 7 | `machine_processing_or_model_training_prohibited` | none identified |
| 8 | `attribution_obligations_executable` | European Union source, CELEX identifier and modification notice |
| 9 | `official_acquisition_reproducible` | official EUR-Lex data dump or Cellar API |
| 10 | `checksum_or_signed_manifest_available` | official dump metadata plus generated SHA-256 required |
| 11 | `language_composition_officially_supported` | {"en": 1.0} |
| 12 | `corpus_v3_relationship_understood` | new structured legal prose; high DGT/JRC overlap risk |
| 13 | `conservative_capacity_estimable_without_download` | broad legal corpus reduced for eligible sectors, rights exclusions, consolidated-version duplication and DGT/JRC overlap |
| 14 | `source_concentration_violation` | False |
| 15 | `benchmark_pii_private_or_contamination_concerns` | {"benchmark_contamination": "low", "boilerplate": "high", "copyrighted_long_form": "medium", "pii": "medium", "private_data": "low", "spam": "low", "unsafe_content": "low"} |

Next action: Pin a CELLAR/data-dump release and preserve document-level reuse notices.

## dergipark_ccby_only_subset_20260722

Final status: **CONDITIONAL**
Resolution path: `bounded_sample_required`
Reason: DergiPark is a platform, not a portal-wide content licence; journals must be allowlisted.

| # | Required question | Recorded answer |
|---:|---|---|
| 1 | `exact_downloadable_artifact` | no approved platform-wide inventory |
| 2 | `exact_snapshot_release_dump_tag_or_commit` | unresolved article-level allowlist and license metadata snapshot |
| 3 | `content_license_explicit` | mixed; admit only explicit CC0, CC BY or separately approved commercial-use terms |
| 4 | `collection_or_database_license_explicit` | platform-wide database/bulk-reuse license not established |
| 5 | `redistribution_rights_explicit` | item specific |
| 6 | `modification_and_derivative_processing_rights_explicit` | item specific |
| 7 | `machine_processing_or_model_training_prohibited` | no platform-wide conclusion; item terms govern |
| 8 | `attribution_obligations_executable` | author, title, journal, DOI/URL, exact license and retrieval snapshot |
| 9 | `official_acquisition_reproducible` | official journal endpoints/OAI only after platform permission confirmation |
| 10 | `checksum_or_signed_manifest_available` | article allowlist and generated SHA-256 required |
| 11 | `language_composition_officially_supported` | {"en": 0.2, "tr": 0.8} |
| 12 | `corpus_v3_relationship_understood` | high-value non-Wikimedia Turkish prose with duplicate and rights risk |
| 13 | `conservative_capacity_estimable_without_download` | broad platform volume reduced for explicit commercial-license allowlisting, PDF quality, references, PII and cross-journal duplicates |
| 14 | `source_concentration_violation` | False |
| 15 | `benchmark_pii_private_or_contamination_concerns` | {"benchmark_contamination": "medium", "boilerplate": "high", "copyrighted_long_form": "high", "pii": "medium", "private_data": "low", "spam": "low", "unsafe_content": "medium"} |

Next action: DergiPark is a platform, not a portal-wide content licence; journals must be allowlisted.

Future bounded sample (not authorized):
maximum 25,000,000 bytes; destination `$EXTERNAL_SSD_ROOT\DarkMindArchive\corpus-v4\samples\dergipark_ccby_only_subset_20260722`; SHA-256 required; training use is false.

## project_gutenberg_turkish_vetted

Final status: **DEFERRED**
Resolution path: `metadata_resolvable`
Reason: Rights and trademark conditions are work- and jurisdiction-specific; no Turkish allowlist is pinned.

| # | Required question | Recorded answer |
|---:|---|---|
| 1 | `exact_downloadable_artifact` | Turkish-language candidate list not pinned |
| 2 | `exact_snapshot_release_dump_tag_or_commit` | unresolved Turkish work-ID allowlist |
| 3 | `content_license_explicit` | works not restricted under U.S. copyright law; local-law review required outside the U.S. |
| 4 | `collection_or_database_license_explicit` | Project Gutenberg terms/trademark conditions apply to collection files |
| 5 | `redistribution_rights_explicit` | work and jurisdiction specific |
| 6 | `modification_and_derivative_processing_rights_explicit` | work specific; Project Gutenberg boilerplate/trademark terms apply |
| 7 | `machine_processing_or_model_training_prohibited` | none identified for verified works |
| 8 | `attribution_obligations_executable` | author, title, ebook ID, source and documented rights determination |
| 9 | `official_acquisition_reproducible` | official catalog/bulk mechanism only; robots policy respected |
| 10 | `checksum_or_signed_manifest_available` | official catalog IDs plus generated SHA-256 |
| 11 | `language_composition_officially_supported` | {"tr": 1.0} |
| 12 | `corpus_v3_relationship_understood` | new book prose with high public-domain mirror overlap risk |
| 13 | `conservative_capacity_estimable_without_download` | small Turkish catalog estimate reduced for jurisdiction review, OCR quality, boilerplate and mirror duplication |
| 14 | `source_concentration_violation` | False |
| 15 | `benchmark_pii_private_or_contamination_concerns` | {"benchmark_contamination": "medium", "boilerplate": "high", "copyrighted_long_form": "high", "pii": "low", "private_data": "low", "spam": "low", "unsafe_content": "medium"} |

Next action: Rights and trademark conditions are work- and jurisdiction-specific; no Turkish allowlist is pinned.

## pmc_oa_commercial_reuse_subset

Final status: **CONDITIONAL**
Resolution path: `metadata_resolvable`
Reason: Admit only exact OA articles carrying CC0, CC BY, CC BY-SA, or separately approved terms.

| # | Required question | Recorded answer |
|---:|---|---|
| 1 | `exact_downloadable_artifact` | commercial-use group exists but must be narrowed to CC0/BY/BY-SA and removed-item status |
| 2 | `exact_snapshot_release_dump_tag_or_commit` | unresolved dated OA file list and removed-items list |
| 3 | `content_license_explicit` | only CC0, CC BY and CC BY-SA admitted; ND, NC and unclear items excluded |
| 4 | `collection_or_database_license_explicit` | PMC OA collection terms require article-level rights compliance |
| 5 | `redistribution_rights_explicit` | permitted only under each admitted article license |
| 6 | `modification_and_derivative_processing_rights_explicit` | permitted for CC0/BY/BY-SA; ND excluded |
| 7 | `machine_processing_or_model_training_prohibited` | none identified for admitted licenses |
| 8 | `attribution_obligations_executable` | PMCID, citation, authors, source URL, license and snapshot |
| 9 | `official_acquisition_reproducible` | PMC Cloud/FTP/OAI/API/BioC approved services only |
| 10 | `checksum_or_signed_manifest_available` | official file list plus per-package hashes where available and generated SHA-256 |
| 11 | `language_composition_officially_supported` | {"en": 0.98, "other_rejected": 0.02} |
| 12 | `corpus_v3_relationship_understood` | new scientific prose with high benchmark and near-duplicate risk |
| 13 | `conservative_capacity_estimable_without_download` | large OA inventory reduced for license allowlist, English filtering, references, XML extraction, retractions, PII and benchmark overlap |
| 14 | `source_concentration_violation` | False |
| 15 | `benchmark_pii_private_or_contamination_concerns` | {"benchmark_contamination": "high", "boilerplate": "high", "copyrighted_long_form": "high", "pii": "medium", "private_data": "medium", "spam": "low", "unsafe_content": "medium"} |

Next action: Admit only exact OA articles carrying CC0, CC BY, CC BY-SA, or separately approved terms.

## openstax_legacy_ccby_editions

Final status: **DEFERRED**
Resolution path: `metadata_resolvable`
Reason: Legacy CC BY editions and embedded third-party exceptions require an edition-level manifest.

| # | Required question | Recorded answer |
|---:|---|---|
| 1 | `exact_downloadable_artifact` | legacy CC BY edition inventory not pinned |
| 2 | `exact_snapshot_release_dump_tag_or_commit` | unresolved immutable edition file list |
| 3 | `content_license_explicit` | legacy CC BY editions only; newer CC BY-NC-SA excluded |
| 4 | `collection_or_database_license_explicit` | not_applicable |
| 5 | `redistribution_rights_explicit` | allowed only for pinned CC BY editions |
| 6 | `modification_and_derivative_processing_rights_explicit` | CC BY adaptations permitted with attribution |
| 7 | `machine_processing_or_model_training_prohibited` | none identified for CC BY editions |
| 8 | `attribution_obligations_executable` | OpenStax, title, edition, contributors, URL and exact notice |
| 9 | `official_acquisition_reproducible` | official OpenStax edition download links |
| 10 | `checksum_or_signed_manifest_available` | edition file SHA-256 and embedded-license capture required |
| 11 | `language_composition_officially_supported` | {"en": 1.0} |
| 12 | `corpus_v3_relationship_understood` | new educational prose with edition duplication risk |
| 13 | `conservative_capacity_estimable_without_download` | legacy catalog estimate reduced for NC-SA transition, edition pinning, duplicated revisions and textbook boilerplate |
| 14 | `source_concentration_violation` | False |
| 15 | `benchmark_pii_private_or_contamination_concerns` | {"benchmark_contamination": "medium", "boilerplate": "medium", "copyrighted_long_form": "medium", "pii": "low", "private_data": "low", "spam": "low", "unsafe_content": "low"} |

Next action: Legacy CC BY editions and embedded third-party exceptions require an edition-level manifest.

## stack_exchange_ccbysa_dump

Final status: **DEFERRED**
Resolution path: `bounded_sample_required`
Reason: Attribution reconstruction, PII, benchmark leakage, and current dump identity remain unresolved.

| # | Required question | Recorded answer |
|---:|---|---|
| 1 | `exact_downloadable_artifact` | site XML archives, exact date and site allowlist unresolved |
| 2 | `exact_snapshot_release_dump_tag_or_commit` | unresolved official dump date and checksums |
| 3 | `content_license_explicit` | CC BY-SA version determined by contribution date and revision history |
| 4 | `collection_or_database_license_explicit` | dump/database redistribution terms require confirmation with attribution graph |
| 5 | `redistribution_rights_explicit` | subject to post-level CC BY-SA and attribution |
| 6 | `modification_and_derivative_processing_rights_explicit` | permitted under applicable CC BY-SA with ShareAlike |
| 7 | `machine_processing_or_model_training_prohibited` | no blanket prohibition identified in content license; platform terms still apply |
| 8 | `attribution_obligations_executable` | post URL, author, title, date and license version |
| 9 | `official_acquisition_reproducible` | officially referenced Stack Exchange data dump only |
| 10 | `checksum_or_signed_manifest_available` | Archive.org file inventory/checksums must be pinned |
| 11 | `language_composition_officially_supported` | {"code": 0.05, "en": 0.95} |
| 12 | `corpus_v3_relationship_understood` | new technical dialogue with high duplication, PII and benchmark risk |
| 13 | `conservative_capacity_estimable_without_download` | technical post volume reduced for site allowlist, PII, signatures, low-quality fragments, duplicate answers, benchmarks and attribution burden |
| 14 | `source_concentration_violation` | False |
| 15 | `benchmark_pii_private_or_contamination_concerns` | {"benchmark_contamination": "high", "boilerplate": "high", "copyrighted_long_form": "medium", "pii": "high", "private_data": "medium", "spam": "high", "unsafe_content": "medium"} |

Next action: Attribution reconstruction, PII, benchmark leakage, and current dump identity remain unresolved.

Future bounded sample (not authorized):
maximum 25,000,000 bytes; destination `$EXTERNAL_SSD_ROOT\DarkMindArchive\corpus-v4\samples\stack_exchange_ccbysa_dump`; SHA-256 required; training use is false.

## github_permissive_repo_allowlist

Final status: **REJECTED**
Resolution path: `policy_rejected`
Reason: Public repository visibility is not a licence and a platform-wide allowlist is not auditable.

| # | Required question | Recorded answer |
|---:|---|---|
| 1 | `exact_downloadable_artifact` | no repository allowlist yet |
| 2 | `exact_snapshot_release_dump_tag_or_commit` | unresolved repository/commit/file allowlist |
| 3 | `content_license_explicit` | only approved permissive licenses with file-level exceptions resolved |
| 4 | `collection_or_database_license_explicit` | not_applicable_to_individual_git_repositories |
| 5 | `redistribution_rights_explicit` | license specific |
| 6 | `modification_and_derivative_processing_rights_explicit` | license specific |
| 7 | `machine_processing_or_model_training_prohibited` | license and repository terms checked per source |
| 8 | `attribution_obligations_executable` | repository, commit, authorship, source URL and required notices |
| 9 | `official_acquisition_reproducible` | official Git endpoint, one approved repository at a time |
| 10 | `checksum_or_signed_manifest_available` | Git commit plus per-file SHA-256 manifest |
| 11 | `language_composition_officially_supported` | {"code": 0.8, "en": 0.2} |
| 12 | `corpus_v3_relationship_understood` | new code; high fork/vendor/benchmark duplication risk |
| 13 | `conservative_capacity_estimable_without_download` | allowlisted code estimate reduced for license exceptions, vendors, generated/minified files, forks, tests, secrets and code-quality filters |
| 14 | `source_concentration_violation` | False |
| 15 | `benchmark_pii_private_or_contamination_concerns` | {"benchmark_contamination": "high", "boilerplate": "medium", "copyrighted_long_form": "medium", "pii": "high", "private_data": "medium", "spam": "medium", "unsafe_content": "medium"} |

Next action: none

## arxiv_explicit_commercial_cc_subset

Final status: **DEFERRED**
Resolution path: `bounded_sample_required`
Reason: Author-selected licences, version identity, long-form overlap, and benchmark leakage need a frozen allowlist.

| # | Required question | Recorded answer |
|---:|---|---|
| 1 | `exact_downloadable_artifact` | full source collection about 9.2 TB as of April 2025; explicit-CC subset not enumerated |
| 2 | `exact_snapshot_release_dump_tag_or_commit` | unresolved OAI license allowlist and S3 manifest version |
| 3 | `content_license_explicit` | CC0, CC BY or CC BY-SA only; default arXiv, NC and ND excluded |
| 4 | `collection_or_database_license_explicit` | bulk S3 terms and manifests apply; article rights remain per item |
| 5 | `redistribution_rights_explicit` | only for explicit admitted licenses |
| 6 | `modification_and_derivative_processing_rights_explicit` | CC0/BY/BY-SA permit adaptation; other licenses excluded |
| 7 | `machine_processing_or_model_training_prohibited` | none identified for admitted licenses |
| 8 | `attribution_obligations_executable` | authors, title, arXiv ID, version, URL and exact license |
| 9 | `official_acquisition_reproducible` | official OAI metadata then requester-pays S3 only |
| 10 | `checksum_or_signed_manifest_available` | official S3 manifests include MD5, size and item count; selected files also SHA-256 |
| 11 | `language_composition_officially_supported` | {"en": 0.98, "other_rejected": 0.02} |
| 12 | `corpus_v3_relationship_understood` | new scientific prose with benchmark and preprint-version duplication risk |
| 13 | `conservative_capacity_estimable_without_download` | large corpus reduced sharply because most papers use the default non-reusable license, then for versions, LaTeX failures and benchmarks |
| 14 | `source_concentration_violation` | False |
| 15 | `benchmark_pii_private_or_contamination_concerns` | {"benchmark_contamination": "high", "boilerplate": "high", "copyrighted_long_form": "high", "pii": "medium", "private_data": "low", "spam": "medium", "unsafe_content": "medium"} |

Next action: Author-selected licences, version identity, long-form overlap, and benchmark leakage need a frozen allowlist.

Future bounded sample (not authorized):
maximum 25,000,000 bytes; destination `$EXTERNAL_SSD_ROOT\DarkMindArchive\corpus-v4\samples\arxiv_explicit_commercial_cc_subset`; SHA-256 required; training use is false.
