# Phase 2A Hugging Face Release Plan

- Planned repository: `petrofi/darkmind-v2-tiny-base`
- Release type: research preview
- Model type: base causal language model
- Instruction-tuned: no
- Conversational assistant: no
- Upload status: no Hugging Face upload has occurred

## Compatibility Approach

The package uses custom Transformers code with `AutoConfig`,
`AutoModelForCausalLM`, and `AutoTokenizer` mappings through
`trust_remote_code=True`. This preserves the actual DarkMind v2 architecture
and frozen SentencePiece behavior. Public weights use safetensors. A local,
ignored fixture export validates offline loading; it is explicitly marked as
untrained and must never be published as a model.

## Intended Use And Limitations

The future artifact is intended for transparent small-model research,
reproducible Turkish/English base-model experiments, and educational pipeline
inspection. It will not be suitable for high-stakes use, factual reliance, or
assistant deployment. The 50M-character pilot corpus has limited domain and
language coverage, and the 256-token smoke architecture is deliberately small.

## Required Release Files

`config.json`, `model.safetensors`, frozen `tokenizer.model` and
`tokenizer.vocab`, tokenizer/special-token config, generation config, custom
model/tokenizer code, model card, training/evaluation metadata, and a
provenance/hash manifest are required.

## Corpus, Tokenizer, And Licensing

The model card must summarize the approved Turkish/English Phase 1B corpus,
link its source-license and attribution records, identify Candidate D BPE 24k,
and publish immutable tokenizer hashes. Redistribution must remain compatible
with every source's license and attribution conditions.

## Release Gates

Release requires an approved base run, decreasing finite validation loss,
checkpoint and tokenizer hash compatibility, fixed-prompt health checks,
human review, complete attribution, safetensors-only public weights, local
offline AutoClass round-trip, and a reproducible export manifest. No SFT or
public upload occurs before these gates pass.
