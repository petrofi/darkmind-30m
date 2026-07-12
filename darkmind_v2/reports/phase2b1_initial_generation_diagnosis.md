# Phase 2B.1 Initial Generation Diagnosis

## Result

**PASS: valid randomly selected normal vocabulary pieces.**

The two initialization warnings do not indicate tokenizer corruption, decoding
corruption, invalid Unicode, byte-fallback corruption, or a model-forward
defect. The zero-token randomly initialized model selected valid SentencePiece
vocabulary entries that contain Cyrillic and CJK characters.

Reproduction used:

- model seed: `20260712`
- model config: `darkmind_v2/config/model_tiny_smoke.json`
- frozen tokenizer: `darkmind_v2_sp_bpe24k_v1`
- fixed prompt set: `darkmind_v2/eval/fixed_base_prompts.jsonl`
- generation: deterministic greedy, 16 new tokens

## en_technical_001

Prompt: `Python is a programming language that`

Complete decoded continuation:

```text
аа筆筆筆筆筆 SırpPointPointPointPointPointPointPointPoint
```

Escaped representation:

```text
\u0430\u0430\u7B46\u7B46\u7B46\u7B46\u7B46 S\u0131rpPointPointPointPointPointPointPointPoint
```

Generated token IDs:

```text
22357, 22357, 23852, 23852, 23852, 23852, 23852, 10212,
16902, 16902, 16902, 16902, 16902, 16902, 16902, 16902
```

SentencePiece pieces:

```text
а | а | 筆 | 筆 | 筆 | 筆 | 筆 | ▁Sırp |
Point | Point | Point | Point | Point | Point | Point | Point
```

The model file reports piece U+7B46 as the normal vocabulary piece represented
above. The decoded character is U+7B46 CJK UNIFIED IDEOGRAPH-7B46.

- normal vocabulary pieces: 16
- byte-fallback pieces: 0
- detected scripts: Cyrillic, CJK/other, Latin
- notable code points: U+0430 CYRILLIC SMALL LETTER A, U+7B46 CJK UNIFIED IDEOGRAPH-7B46, U+0131 LATIN SMALL LETTER DOTLESS I
- token-ID encode/decode round-trip: exact
- decoded-text round-trip: exact
- invalid Unicode / replacement / mojibake: 0 / 0 / 0
- token-range violations: 0
- model logits finite: yes

## tr_technical_002

Prompt: `Bir web sunucusu istemciden gelen`

Complete decoded continuation:

```text
 resimatible rights rights rights rightsPointPointPointPointPointPointЖЖЖЖ
```

Escaped representation:

```text
 resimatible rights rights rights rightsPointPointPointPointPointPoint\u0417\u0417\u0417\u0417
```

Generated token IDs:

```text
3362, 18829, 10114, 10114, 10114, 10114, 16902, 16902,
16902, 16902, 16902, 16902, 22820, 22820, 22820, 22820
```

SentencePiece pieces:

```text
▁resim | atible | ▁rights | ▁rights | ▁rights | ▁rights |
Point | Point | Point | Point | Point | Point | Ж | Ж | Ж | Ж
```

- normal vocabulary pieces: 16
- byte-fallback pieces: 0
- detected scripts: Latin, Cyrillic
- notable code point: U+0417 CYRILLIC CAPITAL LETTER ZE
- token-ID encode/decode round-trip: exact
- decoded-text round-trip: exact
- invalid Unicode / replacement / mojibake: 0 / 0 / 0
- token-range violations: 0
- model logits finite: yes

## Integrity Checks

- tokenizer.model: `db116d4bcf315a6d2a7c5191cbea719d5751c9ba839778eba7e243d520253445`
- tokenizer.vocab: `f098fecdd4f610ce5b150be09e56e7648211e1ecb076ad6f38af71cee25344ed`
- freeze manifest: `8e452c049f05ef1c6a94cb5fb42b6accdd1c18b76edebdb9d68bd85fbdfe538e`
- vocabulary range: every generated ID is in `[0, 23999]`
- decode exceptions: 0
- non-finite logits: 0

## Determination

1. Valid randomly selected vocabulary pieces: **yes**
2. Valid byte-fallback sequences: **no; none were used**
3. Tokenizer corruption: **no**
4. Decoding corruption: **no**
5. Model or pipeline defect: **no**

Unexpected-script and repetition findings are initialization-quality warnings.
They remain visible in metrics, but they are not integrity failures for a
randomly initialized zero-token checkpoint.
