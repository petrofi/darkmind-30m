# DarkMind v2 Tiny Full-Epoch Capacity Diagnostic

This package is a local-only research artifact for the 9,369,088-parameter
DarkMind v2 tiny decoder-only model. It is not publicly released.

## Training scope

- Clean from-scratch initialization; Stage-1 weights were not reused.
- One deterministic epoch-equivalent.
- 11,743,232 train tokens consumed in 2,867 optimizer steps.
- 99.9915% train-token coverage; a deterministic 994-token tail was excluded.
- BF16 training with a 4,096-token effective optimizer batch.
- Frozen 24,000-piece tokenizer with tied input/output embeddings.

## Measured result

- Validation loss: 10.123761 to 5.483633.
- Eval loss: 10.123414 to 5.476849.
- Final perplexity: 240.720.
- Greedy repetition warning rate: 75.5%.
- Greedy exact n-gram loop rate: 74.5%.
- Greedy EOS completion: 25.0%.
- Public-preview mechanical hard failures: 0.

## Limitations

This checkpoint is a tiny capacity diagnostic. It is not instruction-tuned,
not a chat model, not production-ready, and not approved for public release.
Outputs remain largely repetitive, incoherent, contextually weak, and
factually unreliable. Turkish, English, factual, technical, and code quality
are all inadequate for practical use.

The model-weight distribution license is unresolved and remains a release
blocker. No Hugging Face upload is authorized or performed by the export
tooling.
