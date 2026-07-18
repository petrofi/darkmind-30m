# DarkMind v2 Phase 4C Optimizer Group Audit

Result: **PASS with a policy finding**

Unique trainable tensors: 172; elements: 118,056,960.
Current decay coverage: 118,056,960 elements; current no-decay coverage: 0.
Recommended decay coverage: 99,090,432; recommended no-decay coverage: 18,966,528.

The current implementation passes every trainable parameter once to one AdamW group. It therefore applies weight decay 0.1 to biases, LayerNorm weights, token and position embeddings, and the tied input/output embedding storage.

The tied token embedding and LM head are the same parameter object and create exactly one optimizer-state entry. No duplicate state exists.

A corrected-grouping arm is required by the Phase 4C protocol. The proposed policy keeps decay on Linear matrix weights and removes decay from bias, normalization, token embedding, positional embedding, and tied LM-head aliases.

Full per-parameter names, shapes, dtypes, storage identities, aliases, and group assignments are retained in the JSON report.
