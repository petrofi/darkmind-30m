# DarkMind v2 Phase 4B Runtime Storage Inventory

Original Phase 3C runtime: 23,260,871,301 bytes
Minimal copied immutable inputs: 1,298,531,397 bytes
Phase 4A checkpoints: 3,088,600,440 bytes
Total new Phase 4B runtime: 8,117,979,037 bytes

| Phase 4B arm | Bytes |
|---|---:|
| arm_a_legacy_lr3e4 | 1,696,432,706 |
| arm_b_legacy_lr15e5 | 1,698,893,262 |
| arm_c_stratified_lr3e4 | 1,709,604,593 |
| arm_d_stratified_lr15e5 | 1,709,564,116 |

Must not be deleted: original Phase 4A evidence, original Corpus V3 runtime, relocated immutable inputs and validation manifests, factorial summaries, order manifests, raw audits, and storage/factor analysis manifests.

Safe to archive later after an explicit user decision: model-only checkpoints from the four failed exploratory arms. Retain their hashes, summaries, metrics, and raw audit manifests with any archive.

Required for further diagnosis: `inputs/corpus_v3_tokenized`, frozen tokenizer package, Base V1 config, both order manifests, validation passes, factorial configs, and all arm summaries/audits. No checkpoint is approved for 25M continuation.

Recommended external SSD layout: `DarkMindRuntime/phase4b/inputs`, `runs/failed_factorial`, `reports`, `exports`, `logs`, and `archive/phase4a_phase4b_evidence`. Preserve manifests beside every moved archive and verify hashes after any future copy.

No file was deleted, moved, or archived in this task.
