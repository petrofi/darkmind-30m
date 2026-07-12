---
license: other
library_name: transformers
pipeline_tag: text-generation
language:
- tr
- en
tags:
- research
- causal-lm
- base-model
---

# DarkMind v2 Tiny Stage-1

## Status

**Public research-preview audit result: FAIL - additional work is required before public release.**

This is a local research and pipeline-validation checkpoint. It is not instruction-tuned, not a chat model, not production-ready, and not suitable for safety-critical, factual, coding, or user-facing use. It should not be described as fluent, safe, conversational, or instruction-following.

No public Hugging Face upload has occurred.

## Model Details

- Model type: decoder-only causal base language model
- Parameters: 9,369,088
- Layers / attention heads / embedding dimension: 4 / 4 / 256
- Context block size: 256
- Tokenizer: SentencePiece BPE, 24,000 tokens
- Tied input/output embeddings: yes
- Trained tokens: 1,048,576
- Training fraction: approximately 8.9% of one training epoch
- Optimizer steps: 256
- Selected checkpoint: `step_000256_tokens_001048576`
- Checkpoint SHA-256: `5e2fd69d4775940629926a7bf659e36beafe4c1cd544feb8d97beeae6537b097`
- Tokenizer model SHA-256: `db116d4bcf315a6d2a7c5191cbea719d5751c9ba839778eba7e243d520253445`
- Tokenizer vocab SHA-256: `f098fecdd4f610ce5b150be09e56e7648211e1ecb076ad6f38af71cee25344ed`

## Evaluation

- Full validation loss: 7.096028
- Full eval loss: 7.081271
- Perplexity: 1,207.163
- Public-preview greedy audit: 200 outputs
- Public-preview seeded audit: 500 outputs across two profiles and five fixed seeds
- Public-preview generation hard failures: 0
- Invalid UTF-8 / U+FFFD / mojibake outputs: 0 / 0 / 0

Mechanical health does not imply language quality. Greedy evaluation produced repetition warnings in 179/200 outputs and exact repeated n-gram loops in 131/200. The longest repeated-token run was 31. Sampling produced repetition warnings in 158/500 outputs and exact loops in 67/500. All ten greedy code prompts failed a conservative code-continuation check.

Representative outputs are frequently numeric fragments, repeated punctuation or words, very short completions, and incoherent continuations. Factual statements are unverified and unreliable. Script mixing did not occur in this audit, but remains a known risk for a minimally trained multilingual model.

## Training Corpus

The pilot corpus contains approximately 50 million normalized characters: about 30 million Turkish and 20 million English characters. Content includes Turkish Wikipedia, English Wikiversity, Turkish Wikibooks, Turkish Wikivoyage, Turkish and English Python documentation, and Tatoeba sentences.

Source-level license and attribution references are listed in `corpus_attribution.json`. The complete local per-document attribution manifest has SHA-256 `b820ec56a0c173604a5a97663c1ca510c7c78800d3e559722b23ffb74eb3120f` and must be preserved for any redistribution workflow.

## License Information

Repository source code is covered by the repository's MIT license. Training sources retain their own licenses, including CC BY-SA 4.0/GFDL, CC BY 2.0 FR, PSF License v2, and 0BSD example-code terms.

No standalone distribution license has yet been designated for the model weights. This package therefore must not be publicly uploaded or redistributed until the owner selects appropriate model-weight terms and confirms source-license and attribution obligations. See `LICENSE_INFORMATION.md`.

## Intended Uses

- Reproducible causal-LM pipeline validation
- Checkpoint loading and export testing
- Tokenizer and generation-health research
- Controlled local study of very early pretraining behavior

## Out-of-Scope Uses

- Chat or instruction following
- Production deployment
- Factual question answering
- Reliable Turkish or English generation
- Code generation
- Safety-critical decisions
- Automated publication or user-facing content

## Local Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "darkmind-v2-tiny-stage1"
tokenizer = AutoTokenizer.from_pretrained(
    path, trust_remote_code=True, local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, local_files_only=True
)
```

Loading custom code with `trust_remote_code=True` executes files from the model directory. Inspect them before use.

## Limitations

The checkpoint has seen only a small fraction of one epoch. Evaluation prompts are controlled continuation stems and do not comprehensively measure factual accuracy, safety, bias, or downstream usefulness. The 200-prompt greedy set and 500-generation sampling matrix are deterministic mechanical audits, not proof of language understanding. Further base pretraining, a new quality audit, and a model-weight licensing decision are required before public release.
