# Phase 2A Hardware And Training Budget

## Environment Probe

- Python: 3.10.11
- PyTorch: 2.4.1+cu121
- CUDA runtime: 12.1
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- GPU memory: 8,585,216,000 bytes
- BF16 / FP16: supported / supported
- Fused AdamW argument: available and usable
- Logical CPU threads: 20
- PyTorch CPU threads: 14
- Free disk at probe: 298,155,409,408 bytes
- Transformers: 4.55.4
- SentencePiece: 0.2.1
- safetensors: 0.6.2

No required Phase 2A dependency is missing.

## Exact Tiny Fixture Benchmark

- Parameters: 9,369,088
- Device / precision: CUDA / BF16
- Sequence length / micro-batch: 32 / 1
- Steps: 12
- Initial/final loss: 10.0887 / 4.7883
- Mean step time: 0.05882 seconds
- Throughput: 544.03 tokens/second
- Peak CUDA allocation: 209,170,432 bytes
- Final pre-clip gradient norm: 3.8387
- Gradient checkpointing: not used
- Checkpoint reload: PASS
- Restored step: 12
- Generated token range: PASS

A separate four-step exact-model run loaded step 4 and completed a real resumed
optimizer/scheduler update at step 5. Resume continuation therefore passed in
addition to state reload.

A final two-step provenance run stored the actual fixture tokenized-manifest
SHA-256 `de18a61ce831dfc269ecccb815ddc93e73427b1720089e00a5adc30ca0746b7f`
in checkpoint metadata, reloaded step 2, and continued to step 3.

This short fixed-batch result proves that gradients flow and fixture loss can
decrease. It is not a throughput forecast for sequence length 256 and is not a
language-quality result.

## Candidate Profiles For A Future Approved Smoke Run

| Profile | Sequence | Precision | Micro-batch | Accumulation | Effective tokens/update | Fallback |
|---|---:|---|---:|---:|---:|---|
| Safe | 256 | BF16 | 1 | 16 | 4,096 | FP32 micro-batch 1, accumulation 16 |
| Balanced | 256 | BF16 | 4 | 4 | 4,096 | micro-batch 2, accumulation 8 |
| Maximum practical | 256 | BF16 | 8 | 2 | 4,096 | immediately fall back to 4, then 2 on OOM |

The safe profile is the initial recommendation. Balanced and maximum practical
profiles require one-step memory probes at sequence length 256 before approval.
The maximum profile is a ceiling candidate, not a guaranteed setting. Fused
AdamW should remain opt-in until numerical equivalence is checked. Gradient
checkpointing is unnecessary for the tiny model unless an actual sequence-256
probe shows pressure; the future 45M-60M model must be benchmarked separately.

No long training run was started.
