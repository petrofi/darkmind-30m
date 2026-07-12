# Phase 2B.1 Seeded-Sampling Byte Diagnosis

- Result: **PASS**
- Failure category: **5. valid byte-fallback tokens sampled in an invalid UTF-8 order**
- Checkpoint: `darkmind_v2\data\phase2a\runs\tiny_stage1_seed20260712_r2\checkpoints\initial_step_000000`
- Seed: `20260712`
- Tokenizer corruption: **False**
- Decoder defect: **False**

## Proven Root Cause

The initial model sampled valid token 155 (<0x93>) as a one-byte run. 0x93 is not a valid UTF-8 start byte, so SentencePiece emitted U+FFFD.

The raw generated characters are preserved. No token was masked, removed, replaced, or sanitized.

Each affected output contains 15 normal SentencePiece vocabulary pieces followed by one byte-fallback piece. There are no control/special, unknown, or out-of-range tokens. Re-encoding the decoded U+FFFD does not reproduce the original byte token ID, while the decoded-text round trip remains stable; this is expected information loss after replacement decoding. Complete token types, Unicode code points, scripts, detector matches, round-trip details, and escaped/raw outputs are preserved in `evaluations/byte_trace_policy_v1/seeded_sampling_byte_diagnosis.json`.

## Token And Byte Evidence

### tr_ordinary_001

- Prompt: `'Sabah uyand\u0131\u011f\u0131mda pencerenin d\u0131\u015f\u0131nda'`
- Generated token IDs: `[7718, 15684, 7429, 18044, 7666, 13632, 11435, 22231, 3220, 8101, 18285, 7252, 23968, 9001, 2539, 155]`
- Token pieces: `['▁Mey', '▁declaration', 'LS', '▁isolated', '▁fit', 'ladım', 'biyen', 'y', '▁Le', 'ests', '▁closing', '____', '𐎠', '▁döndürür', 'meyi', '<0x93>']`
- Raw decoded output: `' Mey declarationLS isolated fitlad\u0131mbiyeny Leests closing____\U000103a0 d\xf6nd\xfcr\xfcrmeyi\ufffd'`
- Replacement positions: `[75]`
- Byte run: `93` from token `[155]` / `['<0x93>']`
- Strict UTF-8: **FAIL** at byte offset `0` (invalid start byte)
- SentencePiece byte-run decode: `'\ufffd'`
- Logits finite / token range / tokenizer hash: **True / PASS / PASS**

### tr_ordinary_002

- Prompt: `'Bug\xfcn hava biraz serin oldu\u011fu i\xe7in'`
- Generated token IDs: `[7718, 15684, 7429, 18044, 7666, 13632, 11435, 22231, 14523, 8101, 18285, 7252, 7234, 9001, 2539, 155]`
- Token pieces: `['▁Mey', '▁declaration', 'LS', '▁isolated', '▁fit', 'ladım', 'biyen', 'y', '▁İskoçya', 'ests', '▁closing', '____', '▁piş', '▁döndürür', 'meyi', '<0x93>']`
- Raw decoded output: `' Mey declarationLS isolated fitlad\u0131mbiyeny \u0130sko\xe7yaests closing____ pi\u015f d\xf6nd\xfcr\xfcrmeyi\ufffd'`
- Replacement positions: `[83]`
- Byte run: `93` from token `[155]` / `['<0x93>']`
- Strict UTF-8: **FAIL** at byte offset `0` (invalid start byte)
- SentencePiece byte-run decode: `'\ufffd'`
- Logits finite / token range / tokenizer hash: **True / PASS / PASS**

### tr_ordinary_003

- Prompt: `'Bilgisayar\u0131m a\xe7\u0131ld\u0131\u011f\u0131nda ekranda'`
- Generated token IDs: `[7718, 15684, 7429, 18044, 7666, 13632, 11435, 22231, 3220, 8101, 18285, 7252, 7234, 9001, 2539, 155]`
- Token pieces: `['▁Mey', '▁declaration', 'LS', '▁isolated', '▁fit', 'ladım', 'biyen', 'y', '▁Le', 'ests', '▁closing', '____', '▁piş', '▁döndürür', 'meyi', '<0x93>']`
- Raw decoded output: `' Mey declarationLS isolated fitlad\u0131mbiyeny Leests closing____ pi\u015f d\xf6nd\xfcr\xfcrmeyi\ufffd'`
- Replacement positions: `[78]`
- Byte run: `93` from token `[155]` / `['<0x93>']`
- Strict UTF-8: **FAIL** at byte offset `0` (invalid start byte)
- SentencePiece byte-run decode: `'\ufffd'`
- Logits finite / token range / tokenizer hash: **True / PASS / PASS**

### tr_ordinary_004

- Prompt: `'Uzun bir g\xfcn\xfcn sonunda insan'`
- Generated token IDs: `[7718, 15684, 7429, 18044, 13060, 13632, 11435, 22231, 14523, 8101, 18285, 7252, 7234, 9001, 2539, 155]`
- Token pieces: `['▁Mey', '▁declaration', 'LS', '▁isolated', 'üşter', 'ladım', 'biyen', 'y', '▁İskoçya', 'ests', '▁closing', '____', '▁piş', '▁döndürür', 'meyi', '<0x93>']`
- Raw decoded output: `' Mey declarationLS isolated\xfc\u015fterlad\u0131mbiyeny \u0130sko\xe7yaests closing____ pi\u015f d\xf6nd\xfcr\xfcrmeyi\ufffd'`
- Replacement positions: `[84]`
- Byte run: `93` from token `[155]` / `['<0x93>']`
- Strict UTF-8: **FAIL** at byte offset `0` (invalid start byte)
- SentencePiece byte-run decode: `'\ufffd'`
- Logits finite / token range / tokenizer hash: **True / PASS / PASS**

### tr_factual_001

- Prompt: `"T\xfcrkiye'nin ba\u015fkenti"`
- Generated token IDs: `[7718, 15684, 7429, 18044, 13060, 13862, 11435, 22231, 14523, 4712, 18285, 7252, 7234, 9001, 2539, 155]`
- Token pieces: `['▁Mey', '▁declaration', 'LS', '▁isolated', 'üşter', '▁Sor', 'biyen', 'y', '▁İskoçya', '▁Tokyo', '▁closing', '____', '▁piş', '▁döndürür', 'meyi', '<0x93>']`
- Raw decoded output: `' Mey declarationLS isolated\xfc\u015fter Sorbiyeny \u0130sko\xe7ya Tokyo closing____ pi\u015f d\xf6nd\xfcr\xfcrmeyi\ufffd'`
- Replacement positions: `[85]`
- Byte run: `93` from token `[155]` / `['<0x93>']`
- Strict UTF-8: **FAIL** at byte offset `0` (invalid start byte)
- SentencePiece byte-run decode: `'\ufffd'`
- Logits finite / token range / tokenizer hash: **True / PASS / PASS**

## Artifact Findings

- Normal vocabulary pieces scanned: `23735`
- Normal-piece U+FFFD or mojibake issues: `0`
- Tokenizer files are hash-identical to the frozen manifest.
- The event is a generated-byte-sequence quality warning for this Stage-1 checkpoint, not tokenizer corruption.
- The same event remains release-blocking for any future public-release candidate.
- The corpus source text passed strict UTF-8 scanning across 697,715 lines with zero U+FFFD characters; all eight tokenized shards have zero out-of-range IDs.
