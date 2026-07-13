# Phase 3A Architecture Parameter Matrix

All candidates use the frozen 24k tokenizer, tied embeddings, context 512, pre-layer normalization, and head dimension 64. Counts were verified by instantiating the real implementation one candidate at a time.

| ID | Class | Layers | Width | Heads | Exact params | Transformer body | Vocabulary share | BF16 weights | AdamW state |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 60M | 16 | 512 | 8 | 62,989,312 | 50,701,312 | 19.5081% | 120.14 MiB | 480.57 MiB |
| B | 80M | 13 | 640 | 10 | 79,694,720 | 64,334,720 | 19.2735% | 152.01 MiB | 608.02 MiB |
| C | 100M | 12 | 768 | 12 | 103,881,216 | 85,449,216 | 17.7433% | 198.14 MiB | 792.55 MiB |
| D | 120M | 14 | 768 | 12 | 118,056,960 | 99,624,960 | 15.6128% | 225.18 MiB | 900.70 MiB |

## Exact Breakdown

### Candidate A (60M)

- Token embeddings: 12,288,000
- Positional embeddings: 262,144
- Attention (weights and biases): 16,809,984
- MLP (weights and biases): 33,595,392
- Normalization parameters: 33,792
- Normalization plus all bias parameters: 107,520
- LM-head-only parameters: 0 (tied)
- Total: 62,989,312
- Target deviation: +4.9822%
- Estimated inference memory: 151.58 MiB
- Estimated training memory before measurement: 1.23 GiB

### Candidate B (80M)

- Token embeddings: 15,360,000
- Positional embeddings: 327,680
- Attention (weights and biases): 21,332,480
- MLP (weights and biases): 42,640,000
- Normalization parameters: 34,560
- Normalization plus all bias parameters: 109,440
- LM-head-only parameters: 0 (tied)
- Total: 79,694,720
- Target deviation: -0.3816%
- Estimated inference memory: 185.44 MiB
- Estimated training memory before measurement: 1.49 GiB

### Candidate C (100M)

- Token embeddings: 18,432,000
- Positional embeddings: 393,216
- Attention (weights and biases): 28,348,416
- MLP (weights and biases): 56,669,184
- Normalization parameters: 38,400
- Normalization plus all bias parameters: 121,344
- LM-head-only parameters: 0 (tied)
- Total: 103,881,216
- Target deviation: +3.8812%
- Estimated inference memory: 233.58 MiB
- Estimated training memory before measurement: 1.87 GiB

### Candidate D (120M)

- Token embeddings: 18,432,000
- Positional embeddings: 393,216
- Attention (weights and biases): 33,073,152
- MLP (weights and biases): 66,114,048
- Normalization parameters: 44,544
- Normalization plus all bias parameters: 141,312
- LM-head-only parameters: 0 (tied)
- Total: 118,056,960
- Target deviation: -1.6192%
- Estimated inference memory: 260.61 MiB
- Estimated training memory before measurement: 2.13 GiB

## Interpretation

Candidate B uses 13 layers rather than the 14-layer starting shape because the 14-layer implementation exceeds the 80M class tolerance. All vocabulary shares are below the preferred 20% threshold. Memory values are conservative analytical planning estimates, not substitutes for the RTX 4060 measurements.
