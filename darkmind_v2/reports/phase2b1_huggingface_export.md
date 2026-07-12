# Phase 2B.1 Local Hugging Face Export

## Export

- Path: `darkmind_v2/data/phase2a/exports/darkmind-v2-tiny-stage1/`
- Selected checkpoint: `step_000256_tokens_001048576`
- Parameters: `9,369,088`
- Training tokens: `1,048,576`
- Approximate train epoch fraction: `8.9%`
- Public upload: **not performed**

The package contains `config.json`, `model.safetensors`, frozen tokenizer files and configs, remote-code model/tokenizer modules, generation config, model card, training metrics, evaluation results, provenance, and file hashes.

## Offline Validation

| Check | Result |
| --- | --- |
| AutoConfig with `trust_remote_code=True` | PASS |
| AutoModelForCausalLM, local files only | PASS |
| AutoTokenizer, local files only | PASS |
| Safetensors load | PASS |
| Finite forward pass | PASS |
| Greedy generation | PASS |
| Frozen tokenizer hashes | PASS |
| Export file hashes | PASS: 14 files |

The exported safetensors hash is `2443baacefc2a3960c24fdab44b66c9465313ab3009293de3d575ab8098a042c`. The source checkpoint hash remains separately recorded in provenance.

This is a local research checkpoint, not an instruction-tuned model, conversational assistant, production model, or public release candidate. Byte-fallback findings and repetition risks are prominent in the model card. No upload occurred.
