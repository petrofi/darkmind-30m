# DarkMind v2 Hugging Face Export

Phase 2A uses a custom Transformers implementation with `AutoConfig`,
`AutoModelForCausalLM`, and `AutoTokenizer` mappings loaded through
`trust_remote_code=True`. Model weights are stored only as safetensors.

The exporter has no upload operation. Phase 2A may create an ignored local
fixture package to test structure and offline loading, but it must label the
weights as untrained fixture data. Internal optimizer resume files are never
part of a public package.

The planned repository is `petrofi/darkmind-v2-tiny-base`. A real export is
blocked until approved base training and every release gate have passed.
