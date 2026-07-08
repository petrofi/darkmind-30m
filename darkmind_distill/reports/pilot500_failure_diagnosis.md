# Pilot500 TR/EN v2 Failure Diagnosis

No new training was started. No teacher data was generated. Existing checkpoints and datasets were treated as immutable.

## Inputs

- Base checkpoint: `models/darkmind-30m-10k-step15000.pt`
- Student checkpoint: `models/darkmind-30m-qwen-distill-pilot500-tr-en-v2.pt`
- Dataset: `darkmind_distill/data/darkmind_qwen_distill_pilot500_tr_en_v2.jsonl`
- Eval prompts: `darkmind_distill/data/pilot500_tr_en_v2_eval_prompts.jsonl`

## Tokenizer Hashes and Compatibility

- `tokenizer/darkmind-tokenizer/merges.txt`: `c551acb4aa4630fef0df2c9ffdc90118be8bf5d100ee3cf5637d868d27475b7b`
- `tokenizer/darkmind-tokenizer/vocab.json`: `af618a08577c2bd741d3b380f0b32bf3a0ac279caad313735621a17f088e9af7`
- Tokenizer vocabulary size: `5422`
- Base checkpoint vocabulary size: `5422`
- Student checkpoint vocabulary size: `5422`
- Base embedding shape: `[5422, 512]`
- Base LM head shape: `[5422, 512]`
- Student embedding shape: `[5422, 512]`
- Student LM head shape: `[5422, 512]`
- Base input/output weights equal in state dict: `True`
- Student input/output weights equal in state dict: `True`
- Special token IDs: `{'<s>': 1, '<pad>': 0, '</s>': 2, '<unk>': 3, '<mask>': 4, '<|end|>': None}`
- EOS token ID: `2`
- PAD token ID: `0`
- UNK token ID: `3`
- Token IDs within base checkpoint vocabulary bounds: `True`
- Token IDs within student checkpoint vocabulary bounds: `True`

Conclusion: there is no direct vocabulary-size mismatch and no out-of-bounds token ID evidence. Diagnosis A is not confirmed by the compatibility audit.

## Suspicious Vocabulary Analysis

Vocabulary script counts:

- Turkish Latin tokens: `10`
- English/ASCII tokens: `1623`
- Hebrew tokens: `0`
- Greek tokens: `0`
- Arabic tokens: `0`
- Cyrillic tokens: `0`
- Devanagari tokens: `0`
- Japanese/CJK tokens: `0`
- Replacement-character or malformed-looking tokens: `1843`

Suspicious examples include:

- `131`: `Â`
- `132`: `Ã`
- `133`: `Ä`
- `134`: `Å`
- `177`: `ð`
- `261`: `Ä±`
- `268`: `Ã¼`
- `297`: `ullanÄ±cÄ±`
- `302`: `KullanÄ±cÄ±`

Conclusion: the tokenizer can round-trip text, but the vocabulary is heavily byte-level/mojibake-looking. This makes invalid-looking partial byte sequences easy for a weak model to emit. Diagnosis B is confirmed as a contributing cause.

## Round-Trip Results

All fixed round-trip tests matched exactly after encode/decode:

- `Merhaba, sen kimsin?`: exact match `True`, tokens/char `0.25`
- `Python kullanarak küçük bir REST servisini nasıl başlatırım?`: exact match `True`, tokens/char `0.3`
- `Docker konteynerim hemen kapanıyor.`: exact match `True`, tokens/char `0.4286`
- `Validation loss neden yükselir?`: exact match `True`, tokens/char `0.2258`
- `Hello, how do I create a REST API?`: exact match `True`, tokens/char `0.5882`
- `A short Python function returns a list.`: exact match `True`, tokens/char `0.3333`

20 deterministic training prompt/response round trips also matched exactly. This argues against a simple tokenizer/checkpoint ID mismatch.

## Base Greedy Outputs

Greedy decoding used argmax only, no sampling, max_new_tokens `80`, device `cuda`.

Base summary: failures `8/8`; true flags included `mixed_or_foreign_script=2`, `gibberish=8`, `language_match=6`.

Representative base outputs:

- `Merhaba.` -> Hebrew-looking repeated bytes ending with replacement text: `×××...ï¿½`
- `Türkiye hakkında iki cümle yaz.` -> repeated phrase: `TÃ¼rkiye'de, TÃ¼rkiye'de, ...`
- `Python nedir?` -> repeated lines: `Veri:\nVeri:\nVeri:...`
- `Docker nedir?` -> repeated: `Docker!\nDocker!\nDocker!...`
- `Sen kimsin?` -> Hebrew-looking repeated bytes: `×›×›×›...ï¿½`
- `Hello.` -> repeated: `Hello, Hello, Hello, ...`
- `What is Python?` -> degenerate English: `What is was well with the University of Christian Christian...`
- `Write one English sentence.` -> repeated: `The English English English...`

Conclusion: the base checkpoint itself is not reliably language-capable under deterministic decoding.

## Student Greedy Outputs

Student summary: failures `7/8`; true flags included `mixed_or_foreign_script=4`, `gibberish=7`, `language_match=2`, `prompt_relevant=1`.

Representative student outputs:

- `Merhaba.` -> Hebrew-looking repeated bytes: `` `×•×•×•... ``
- `Türkiye hakkında iki cümle yaz.` -> same repeated `TÃ¼rkiye'de` pattern as base
- `Python nedir?` -> Hebrew-looking repeated bytes
- `Docker nedir?` -> `Docker!` only; too shallow to count as a useful answer
- `Sen kimsin?` -> Hebrew-looking repeated bytes
- `Hello.` -> `Hellows, University, University, ...`
- `What is Python?` -> Hebrew-looking repeated bytes
- `Write one English sentence.` -> repeated: `` `back', `back', ... ``

Conclusion: the student is corrupted, but the base already exhibits corruption and degenerate repetition. The student appears to sharpen or redirect some bad modes, especially mixed-script output, rather than being the only source of failure.

## SFT Label-Mask Audit

Result: `PASS` on 20 deterministic samples.

Strict checks:

- Prompt tokens are masked.
- Assistant response labels are supervised.
- EOS is supervised.
- Labels and inputs have equal lengths.
- No off-by-one alignment failure was detected.
- Response labels align with the intended response.

Conclusion: Diagnosis D is not confirmed by the sampled SFT encoding audit.

## Module Weight Deltas

Focus module deltas from base to student:

| Module | Abs Delta Norm | Relative Delta Norm | Max Abs Change |
|---|---:|---:|---:|
| token_embedding | 2.999642 | 0.024026 | 0.004553 |
| position_embedding | 0.135992 | 0.010643 | 0.002111 |
| first_transformer_block | 1.000950 | 0.005068 | 0.003297 |
| final_transformer_block | 1.267133 | 0.006221 | 0.003471 |
| final_layer_norm | 0.036755 | 0.001437 | 0.002479 |
| lm_head | 2.999642 | 0.024026 | 0.004553 |

The token embedding / LM head changed more, relatively, than transformer blocks. Because the base was already unstable, this does not prove SFT alone caused the failure, but it is consistent with full fine-tuning amplifying fragile token distributions.

## Corrected Diagnostic Evaluator Findings

The diagnostic evaluator fails outputs if they contain mixed scripts, replacement characters, degenerate repetition, empty/malformed text, language mismatch, or identity leakage. It does not award success merely because a keyword is present.

Findings:

- Base checkpoint: `8/8` greedy prompts failed.
- Student checkpoint: `7/8` greedy prompts failed.
- Non-identity prompts did not receive identity credit.
- Mixed-script outputs were marked as gibberish and not relevant.
- `Docker!` was the only student output with `prompt_relevant=True`, but it is still not a meaningful assistant answer.

## Primary Diagnosis

Primary diagnosis: **F. Multiple confirmed causes**

Confirmed causes:

1. **C. Base checkpoint is not sufficiently pretrained.** Direct evidence: base greedy diagnostic failed `8/8` prompts, including basic Turkish and English prompts.
2. **B. Poor tokenizer vocabulary design / fragile byte-level vocabulary for this model scale.** Direct evidence: `1843` malformed-looking vocabulary entries, many Turkish tokens represented as mojibake-looking byte-level fragments, and generated outputs that repeatedly emit partial-byte-looking Hebrew/replacement sequences.
3. **E. Full fine-tuning likely amplified instability.** Direct evidence: student has more mixed-script failures than base (`4` vs `2`) and embedding/LM head relative deltas (`0.024026`) are higher than transformer block deltas, but this is secondary because the base was already failing.

Not confirmed:

- **A. Tokenizer/checkpoint mismatch**: vocab sizes, embedding shapes, LM head shapes, special IDs, and token bounds are compatible.
- **D. SFT formatting or label-mask bug**: sampled SFT audit passed prompt masking, response supervision, EOS supervision, length equality, and alignment checks.

## Recommended Next Engineering Path

Recommended path: **Path 4**

Stop SFT on the current base. Plan a new DarkMind generation with:

- a clean 16k-32k Turkish-English tokenizer
- tokenizer version manifest and immutable tokenizer hash tracking
- substantially more clean base-pretraining tokens
- periodic fixed-prompt generation evaluation during base training
- a larger model only after the base pipeline produces readable deterministic outputs

Do not continue from `models/darkmind-30m-qwen-distill-pilot500-tr-en-v2.pt`.
Do not generate more teacher data for the current base.
Do not start new training until the base-pretraining/tokenizer plan is rebuilt.
