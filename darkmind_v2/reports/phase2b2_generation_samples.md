# Phase 2B.2 Representative Generation Samples

## Selection Method

The sample is deterministic and was selected by prompt ID before reviewing output quality: the first five Turkish ordinary prompts, first five English ordinary prompts, and fixed technical/code IDs, each with greedy and/or Profile A sampling. It is not a best-of selection.

- Greedy: max 32 new tokens, EOS-aware, no sampling.
- Profile A: temperature 0.7, top-p 0.9, top-k 40, seed 20260712, max 32 new tokens, EOS-aware.
- Outputs are complete and unsanitized.
- `incoherent_text` below is a manual-review warning and is not included in the mechanical manifest totals.

## Turkish Samples

| Mode | Prompt ID | Raw output | Automatic warnings | Manual assessment |
| --- | --- | --- | --- | --- |
| greedy | tr_ordinary_001 | `, 199999999999999999999999999999` | repetition | Weak: numeric loop; incoherent_text |
| greedy | tr_ordinary_002 | `, 191999999999999999999999999999` | repetition | Weak: numeric loop; incoherent_text |
| greedy | tr_ordinary_003 | ` bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir bir,,. 199919999` | repetition | Weak: lexical and numeric loop; incoherent_text |
| greedy | tr_ordinary_004 | `, 199999999999999999999999999999` | repetition | Weak: numeric loop; incoherent_text |
| greedy | tr_ordinary_005 | `, 191999999999999999999999999999` | repetition | Weak: numeric loop; incoherent_text |
| Profile A | tr_ordinary_001 | `, * 11201938950.` | none | Mechanically stronger: EOS and more diversity; still incoherent_text |
| Profile A | tr_ordinary_002 | `, * 11201938950.` | none | Mechanically stronger; unrelated numeric fragment; incoherent_text |
| Profile A | tr_ordinary_003 | ` 2000201.` | repetition | Weak: repeated numeric fragment; incoherent_text |
| Profile A | tr_ordinary_004 | `, * 11201938950.` | none | Mechanically stronger; semantically unusable; incoherent_text |
| Profile A | tr_ordinary_005 | `, * 11201938950.` | none | Mechanically stronger; semantically unusable; incoherent_text |

## English Samples

| Mode | Prompt ID | Raw output | Automatic warnings | Manual assessment |
| --- | --- | --- | --- | --- |
| greedy | en_ordinary_001 | ` the "199999999999999999999999999999` | repetition | Weak: numeric loop; incoherent_text |
| greedy | en_ordinary_002 | ` the "199999999999999999999999999999` | repetition | Weak: numeric loop; incoherent_text |
| greedy | en_ordinary_003 | ` the " " " " " " " " " " " " " " " " " " "1.` | repetition | Weak: punctuation loop; incoherent_text |
| greedy | en_ordinary_004 | ` the " " " " " " " " " " " " " " " " " "1.` | repetition | Weak: punctuation loop; incoherent_text |
| greedy | en_ordinary_005 | ` the " " " " " " " " " " " " " " " " "1.` | repetition | Weak: punctuation loop; incoherent_text |
| Profile A | en_ordinary_001 | ` to be a you.` | none | Mechanically stronger: EOS, unique tokens; ungrammatical/incoherent_text |
| Profile A | en_ordinary_002 | ` to be a you.` | none | Mechanically stronger; ungrammatical/incoherent_text |
| Profile A | en_ordinary_003 | ` to be a of the be "1.` | none | Diverse but incoherent_text |
| Profile A | en_ordinary_004 | ` to be a of the be "1.` | none | Diverse but incoherent_text |
| Profile A | en_ordinary_005 | ` to be as of the be "1.` | none | Diverse but incoherent_text |

## Technical And Code Samples

| Mode | Prompt ID | Raw output | Automatic warnings | Manual assessment |
| --- | --- | --- | --- | --- |
| greedy | tr_technical_001 | `.` | repetition; very_short_output | Weak: no technical continuation; incoherent_text |
| greedy | en_technical_001 | ` the the the the the the the the the the a a a a a a a a a a a a a a a a a a a a a a` | repetition | Weak: token loop; no technical content |
| greedy | code_structured_001 | ` 1919191919191991991919199919991` | code_generation_failure; repetition | Weak: not executable code |
| Profile A | tr_technical_001 | `, * 11201.` | none | Mechanically cleaner; no technical content; incoherent_text |
| Profile A | code_structured_001 | ` 2000201.` | code_generation_failure; repetition | Weak: not executable code |

## Review Conclusion

Sampling reduces long token runs and often reaches EOS, but neither mode demonstrates reliable Turkish, English, factual, technical, or code continuation. The examples support a pipeline checkpoint claim only. They do not support fluency, usefulness, factual reliability, or public model quality claims.
