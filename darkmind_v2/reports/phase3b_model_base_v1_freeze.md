# DarkMind v2 Model Base v1 Freeze

Status: **FROZEN**

Architecture: **darkmind-v2-base-v1 (Candidate D)**

- Total parameters: 118,056,960
- Transformer body: 99,624,960
- Vocabulary share: 15.6128%
- Layers / heads / head dimension: 14 / 12 / 64
- Embedding / MLP hidden dimension: 768 / 3072
- Context length: 512
- Embeddings: tied and immutable
- Attention: SDPA production policy
- Gradient checkpointing: disabled for base v1 production runs
- Precision: BF16
- Tokenizer: `darkmind_v2_sp_bpe24k_v1`
- Tokenizer model SHA-256: `db116d4bcf315a6d2a7c5191cbea719d5751c9ba839778eba7e243d520253445`
- Tokenizer vocab SHA-256: `f098fecdd4f610ce5b150be09e56e7648211e1ecb076ad6f38af71cee25344ed`
- Tokenizer freeze manifest SHA-256: `8e452c049f05ef1c6a94cb5fb42b6accdd1c18b76edebdb9d68bd85fbdfe538e`
- Config SHA-256: `8e9775721b0173a92e88de15c2195428932b3aa5beec57d568674c25887c5e39`
- Architecture hash: `3a2dda86293ceae23ca4e50ea47c840b7fc46021d293c862d330110851ac8305`

## Evidence

Candidate D passed LR calibration, a 4,997,120-token equal-data pilot, a true midpoint process restart, safetensors integrity, post-optimizer memory audit, and a 1,000-step soak. Candidate C learned about 0.44% better by final loss but failed the repeated non-checkpointing process-crash hard gate.

## Limitations and Versioning

The frozen architecture is a 100M-class from-scratch research base. The pilot checkpoint is not a finished model and does not support a conversational-quality claim. The recorded C instability is specific to the observed Windows/PyTorch 2.4.1+cu121 environment; D is the production profile for that environment.

All immutable fields in `model_base_v1_constraints.json` require a new architecture version if changed. A tokenizer, vocabulary, context/position, numerical-backend compatibility, or material architecture change must create v2 rather than mutating base v1.
