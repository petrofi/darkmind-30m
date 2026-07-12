# Phase 2A Tiny Architecture

## Purpose

The Phase 2A model is a decoder-only causal language model used to validate the
DarkMind v2 tokenizer, token shards, forward/backward path, checkpoint resume,
evaluation, and local export. It is not expected to be a strong conversational
model and it is not instruction-tuned.

## Exact Configuration

- Vocabulary: 24,000 frozen SentencePiece BPE tokens
- Context: 256 learned positions
- Transformer blocks: 4
- Attention heads: 4
- Embedding width: 256
- MLP expansion: 4x
- Architecture: pre-layer normalization, GELU, causal self-attention
- Dropout: 0.0 for deterministic smoke validation
- Linear and LayerNorm bias: enabled
- Input/output embeddings: tied and mandatory
- KV cache: deferred until the non-cached path is stable and generation profiling justifies it

## Parameter Breakdown

| Component | Parameters |
|---|---:|
| Token embedding / tied LM head | 6,144,000 |
| Learned position embedding | 65,536 |
| Attention projections, 4 layers | 1,052,672 |
| MLP projections, 4 layers | 2,102,272 |
| Layer normalization | 4,608 |
| Separate LM head | 0 |
| **Total** | **9,369,088** |

Vocabulary-related parameters are 6,144,000, or 65.5774% of the total. An
untied head would add 6,168,000 parameters and is rejected by config
validation.

## Memory Considerations

FP32 parameters alone require about 35.7 MiB; gradients require another 35.7
MiB and AdamW states are substantially larger. Activations scale with batch
size, sequence length, layer count, and width. At sequence length 256 the
attention score tensors are still modest, but profile results rather than a
guess must determine any approved micro-batch. Mixed precision is optional and
auto-detected; CPU FP32 remains a supported fallback.

## Scope And Limitations

The future 45M-60M base model will use more layers and width, likely a longer
context, and a separately approved optimization budget. The 256-token tiny
context is intentionally restrictive. A 50M-character Turkish/English pilot
corpus is adequate for tokenizer and pipeline validation but is too small and
narrow to establish broad factual coverage, robust reasoning, or assistant
quality. Fixture memorization is not model quality.

## v1 Reuse Decision

Reusable concepts are pre-layer-normalized decoder blocks, causal masking, the
fixed prompt categories, and generic Unicode/mojibake checks. The v1
ByteLevelBPE tokenizer, v1 checkpoint dictionary, unshifted loss behavior,
full-file token loading, and v1 training/evaluation launchers are rejected as
incompatible with the frozen v2 SentencePiece tokenizer and v2 provenance
requirements.
