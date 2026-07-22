# Phase 5A external-SSD archival plan

## Boundary and observed total

This is a copy-and-verify plan only. No file was copied, moved, archived, or deleted. All Phase 4B-4F evidence remains in place.

The observed storage subtotal is **73,036,217,908 bytes** (about 68.01 GiB): 29,557,595,581 bytes under `C:\DarkMindRuntime\phase4b` through `phase5a`, 38,295,016,110 bytes of measured worktree data directories, and 5,183,606,217 bytes of measured virtual-environment/cache material. The subtotal excludes ordinary tracked source files and may include intentional deterministic duplicates.

## Git worktrees

| Worktree | Branch | HEAD | Tracked status | Unique ignored/runtime data observed | Disposition |
| --- | --- | --- | --- | ---: | --- |
| `C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-30m` | `codex/self-improvement-pipeline` | `b59ebf8` | 12 unrelated `darkmind_distill`/training changes | `.venv` 4,948,805,335 bytes; caches 229,476,294 bytes | Preserve; unrelated dirty original checkout. |
| `C:\tmp\darkmind-pr5-merge` | `codex/pr5-merge-resolution` | `68c5d1b` | clean | none measured | Old clean merge worktree; archive/removal review only after branch verification. |
| `C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-v2-phase1` | `codex/darkmind-v2-phase1-corpus-tokenizer` | `da90d39` | clean | caches 101,982 bytes | Old Phase 1 worktree; low-value cache only. |
| `C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-v2-phase1b` | `codex/darkmind-v2-phase1b-tokenizer-pilot` | `7e5eb3a` | untracked `WORKTREE_MAP.md` | `darkmind_v2\data` 1,120,418,407 bytes; caches 724,017 bytes | Preserve map and raw/tokenizer evidence until verified archive. |
| `C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-v2-phase2a` | `codex/darkmind-v2-phase2c-tiny-full-epoch` | `ecfa7fd` | six modified frozen-tokenizer metadata files | `darkmind_v2\data` 2,982,224,334 bytes; caches 1,491,829 bytes | Preserve exactly; inspect tokenizer metadata changes separately before any cleanup. |
| `C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-v2-phase3a` | `codex/darkmind-v2-phase3b-finalist-pilots` | `ffe23f4` | clean | `darkmind_v2\data` 7,558,135,811 bytes; caches 764,763 bytes | Archive candidate after hashes and branch provenance are recorded. |
| `C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-v2-phase3c` | `codex/darkmind-v2-phase5a-base-review-corpus-v4-plan` | `790059a` | only untracked Phase 5A planning files | `darkmind_v2\data` 26,634,237,558 bytes; caches 2,241,997 bytes | Active Phase 5A worktree; must remain local. |

Phase 4F source commit `2242284358584cc33148f0867bead49d0caf93be` is contained in `origin/main` by two-parent merge commit `790059a4f61becc47b2b7398b7bbb9b6120d6225` (PR #20).

## Runtime directories

The classifications below are proposals. `archive` means copy while retaining the source. A yes in the delete-after-archive column means only that a later, separately approved cleanup could be considered after full verification.

| Absolute path | Bytes | Purpose | Repro | Future training | Archive | Delete only after verified archive | Must remain local | Depends on |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `C:\DarkMindRuntime\phase4b` | 8,117,998,896 | learning diagnosis inputs and runs | yes | no | yes | candidate | no | Corpus V3, Base V1 |
| `C:\DarkMindRuntime\phase4c` | 12,767,934,680 | optimizer/schedule factorial evidence | yes | no | yes | candidate | no | phase4b inputs |
| `C:\DarkMindRuntime\phase4d` | 3,602,934,431 | stable 25M training evidence and export | yes | historical | yes | selected exploratory material only | no | phase4c decision |
| `C:\DarkMindRuntime\phase4e` | 2,395,182,990 | conditional continuation evidence | yes | historical | yes | candidate | no | phase4d checkpoint |
| `C:\DarkMindRuntime\phase4f` | 2,639,419,028 | final first-pass checkpoint, audits, export | yes | **yes** | yes | **no** | **yes** | phase4d/4e provenance |
| `C:\DarkMindRuntime\phase5a` | 34,125,556 | immutable preflight, 880 raw evaluations, manual packet | yes | evaluation only | yes | no during Phase 5A | yes | phase4f final checkpoint |
| `C:\DarkMindRuntime\phase4f\runs` | 2,398,741,538 | final resume-capable checkpoint and run evidence | yes | **yes** | yes | **no** | **yes** | final Corpus V3 shards |
| `C:\DarkMindRuntime\phase4f\exports` | 240,663,653 | local model-only Transformers export | yes | no | yes | no until release review | yes | phase4f final model |
| `C:\DarkMindRuntime\phase4d\temporary` | 712,193,172 | exploratory intermediate evidence | limited | no | yes | candidate | no | phase4d reports |

## Corpus and worktree data

| Absolute path | Bytes | Purpose | Repro | Future training | Archive | Delete only after verified archive | Must remain local | Depends on |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| `C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-v2-phase3c\darkmind_v2\data\phase3c\raw` | 16,813,074 | early official source archives and checksum manifests | yes | no | yes | candidate | no | official URLs/licenses |
| `C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-v2-phase3c\darkmind_v2\data\phase3c1\raw` | 1,809,106,136 | Corpus V3 raw Wikimedia archives and metadata | yes | rebuild only | yes | candidate | no | source manifests |
| `C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-v2-phase3c\darkmind_v2\data\phase3c1\wikipedia_inventory` | 2,151,021,438 | extracted, normalized, deduplicated Wikipedia inventory | yes | rebuild only | yes | candidate | phase3c1 raw |
| `C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-v2-phase3c\darkmind_v2\data\phase3c1\supplemental_inventory_retry1` | 2,713,629,862 | accepted supplemental extracted/normalized/deduplicated data | yes | rebuild only | yes | candidate | phase3c1 raw and seed |
| `C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-v2-phase3c\darkmind_v2\data\phase3c1\final_text_retry1` | 2,088,747,303 | final deterministic Corpus V3 text | yes | retokenization | yes | no during planning | yes | accepted inventories/allocation |
| `C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-v2-phase3c\darkmind_v2\data\phase3c1\determinism_rebuild` | 9,453,717,889 | independent deterministic rebuild copy | yes | no | yes | candidate after cross-hash | no | same raw inputs/config |
| `C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-v2-phase3c\darkmind_v2\data\phase3c1\tokenized\tranche1_v2` | 2,464,705,227 | final deterministic uint16 shards | yes | **yes** | yes | **no** | **yes** | frozen tokenizer/final text |
| `C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-v2-phase3c\darkmind_v2\data\phase4a` | 3,373,366,257 | failed Phase 4A calibration/run/export evidence | yes | no | yes | candidate | no | Corpus V3/Base V1 |
| `C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-v2-phase1b\darkmind_v2\data` | 1,120,418,407 | Phase 1B raw/tokenizer-pilot data | yes | tokenizer provenance | yes | candidate | no after verification | Phase 1B manifests |
| `C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-v2-phase2a\darkmind_v2\data` | 2,982,224,334 | Phase 2 runtime data | historical | no | yes | candidate only after dirty-worktree audit | no | Phase 2 branch |
| `C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-v2-phase3a\darkmind_v2\data` | 7,558,135,811 | Phase 3 architecture pilot data | yes | no | yes | candidate | no | Phase 3A/3B reports |

Python caches are reproducible and virtual environments can normally be recreated from pinned dependencies, but the original checkout is dirty and is outside Phase 5A scope. Its 4,948,805,335-byte `.venv` and all measured caches remain untouched. No cache cleanup is authorized here.

## Proposed SSD layout

```text
DarkMindArchive/
  immutable-inputs/
    tokenizer/
    configs/
  corpus-v3/
    source-archives/
    normalized/
    final-text/
    tokenized-shards/
    deterministic-rebuild/
  checkpoints/
    base-v1-first-pass-98m/
    historical-gates/
  exports/
    base-v1-first-pass-98m/
  phase-evidence/
    phase4b/
    phase4c/
    phase4d/
    phase4e/
    phase4f/
    phase5a/
  manifests/
  hashes/
```

Use a filesystem that supports large files and preserve timestamps where practical. Keep at least two independent copies of the final checkpoint, frozen tokenizer, final tokenized shards, manifests, and hash records.

## Copy-and-verify procedure

1. Copy one explicitly approved directory to its mapped SSD destination without changing or removing the source.
2. Produce sorted source and destination manifests containing relative path, byte size, and SHA-256 for every file.
3. Compare file counts, total bytes, relative paths, and every hash; any mismatch fails the archive gate.
4. Reopen representative uint16 shards and the final safetensors directly from the SSD; run the offline export-load check against the SSD copy.
5. Record SSD volume identity, destination path, manifest hashes, verification date, and dependencies in an ignored local path map.
6. Only after two verified copies exist may a separate cleanup proposal identify nonessential local candidates. Critical training inputs and the final resume checkpoint stay local until the next phase no longer needs them.
7. Never delete automatically. Any later deletion requires explicit approval, a fresh hash comparison, and a rollback path.

No archival operation was executed in Phase 5A.
