# DarkMind v2 Phase 4B and Phase 4C Runtime Storage Inventory

Phase 4B runtime preserved at `C:\DarkMindRuntime\phase4b`: 8,117,998,896 bytes.
Phase 4C runtime at `C:\DarkMindRuntime\phase4c`: 12,767,934,680 bytes.
Immutable input bytes shared from Phase 4B rather than duplicated: 1,298,531,397 bytes.
Phase 4C copied immutable input bytes: 0 bytes.

| Phase 4C run | Bytes |
|---|---:|
| arm1_global_lr1e4_current_groups | 1,925,296,541 |
| arm2_global_lr75e6_current_groups | 1,925,440,004 |
| arm3_global_lr1e4_corrected_groups | 1,939,876,565 |
| arm4_staged_decay_corrected_groups | 1,925,443,501 |
| arm5_depth_scaled_init_staged | 1,925,467,124 |
| base_v1_stage1_5m_v2_confirmation | 3,124,594,336 |

Resume-capable confirmation checkpoint: `C:\DarkMindRuntime\phase4c\runs\base_v1_stage1_5m_v2_confirmation\checkpoints\step_000610_tokens_004997120` (712,193,172 bytes).

Required for a future 25M continuation: the frozen V2 config, Phase 4B immutable inputs, deterministic order manifest, final step-610 model, optimizer/scheduler/RNG resume state, checkpoint metadata, validation manifests, and hash reports.

Safe to move to an external SSD only after a user-approved copy and hash verification: completed exploratory model-only checkpoints, raw generation manifests, diagnostics, and older Phase 4A/4B evidence. Keep summaries and hashes beside archives.

Safe to delete only after hash-verified archival and explicit approval: redundant exploratory model-only checkpoint payloads. Never delete the frozen inputs, final confirmation resume checkpoint, manifests, or reports before a validated archive exists.

No file was deleted, moved, or archived in Phase 4C.
