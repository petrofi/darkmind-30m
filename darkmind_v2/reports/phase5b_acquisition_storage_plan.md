# Phase 5B acquisition and external-SSD storage plan

## Boundary

No directory was created on an external SSD and no file was copied, moved, archived or downloaded. The plan uses the symbolic root $EXTERNAL_SSD_ROOT; a real volume path, capacity check and operator approval are required in a later phase.

## Proposed layout

| Path under DarkMindArchive/corpus-v4/ | Policy | Regenerable | Needed locally during training |
| --- | --- | --- | --- |
| manifests/ | Immutable, versioned | No | Yes, small |
| source-licenses/ | Immutable, append-only evidence | No | Yes, small |
| raw/ | Immutable after checksum verification | No | No |
| checksums/ | Immutable, append-only | No | Yes, small |
| extracted/ | Append-only per manifest revision | Yes from raw | No |
| normalized/ | Append-only per pipeline version | Yes | No |
| rejected/ | Append-only reason/provenance records; source text minimized | Mostly | No |
| deduplicated/ | Immutable per build identity | Yes | No |
| tokenized/ | Immutable training shards | Yes, expensive | Yes, place on fastest reliable drive |
| attribution/ | Immutable per build identity | No | Yes, small |
| reports/ | Append-only | Yes from manifests where possible | No |
| temporary/ | Temporary, never authoritative | Yes | No |

Repository scripts and Git history stay on the internal drive. During active training, tokenized shards use the fastest reliable drive and checkpoints remain in C:\DarkMindRuntime. Hash-verified checkpoint archives may be planned for the external SSD only in a separately authorized phase.

## Capacity projection

- Current three-source approved raw range: approximately **0.27-1.22 GB**.
- Full future 200M candidate acquisition after source resolution: reserve **up to 0.9 TB raw** until measured inventories replace Phase 5A upper estimates.
- Extracted, normalized, rejection evidence, dedup indexes and rebuild copies: reserve **about 1.1-1.4 TB additional working space** in the worst staged case.
- Final 200M uint16 token IDs: **400 MB** payload; allow **0.5-0.8 GB** with shard indexes and manifests.
- Attribution, license, checksums and reports: tens to hundreds of MB depending document count.

A reliable 2 TB external SSD is the minimum sensible planning size if raw and major intermediates coexist. At least 20% free space must remain before any authorized stage. Acquisition is source-by-source, one worker, smallest approved subset first; no blind bulk retrieval.

## Reproducibility rules

Each entry must match the versioned acquisition schema, approved registry state, official URL, filename, byte range, checksum, license, attribution, retry/rate policy and authorized root. URL or checksum changes require a manifest revision. Raw files become immutable only after checksum verification. The Phase 5B validator contains no downloader.
