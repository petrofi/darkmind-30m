# DarkMind v2 Fixed Base Prompts

`fixed_base_prompts.jsonl` is a deterministic base-model continuation suite, not an instruction-following benchmark.

The prompts are short continuations designed to expose base-model failures:

- empty output,
- repetition,
- mixed-script generation,
- encoding corruption,
- language mismatch,
- broken technical prose,
- invalid code continuation.

Do not add assistant-format prompts such as `Explain...` or `How do I...` to this suite. Those belong to instruction tuning after the base model passes its gates.

