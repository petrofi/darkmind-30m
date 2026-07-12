# Phase 2B.2 Local Release-Package Validation

## Technical Result

**PASS** for local package integrity and offline loading. **NOT ELIGIBLE** for public distribution in the current audit.

| Check | Result |
| --- | --- |
| Required package files | PASS |
| Safetensors-only weights | PASS |
| Pickle / `.bin` / `.pt` model weights absent | PASS |
| Architecture config consistency | PASS |
| Special-token consistency | PASS |
| Frozen tokenizer hashes | PASS |
| Selected checkpoint provenance hash | PASS |
| Final package file hashes | PASS: 16 files |
| Model-card disclosures | PASS |
| Corpus attribution references | PASS: 7 source groups |
| License information present | PASS |
| AutoConfig, local files only | PASS |
| AutoModelForCausalLM, local files only | PASS |
| AutoTokenizer, local files only | PASS |
| Finite forward pass | PASS |
| Greedy generation | PASS |
| Profile A seeded generation | PASS, zero hard failures |
| Network required | No |

Offline seeded output was preserved as ` and the a have to the the of be and the " the you.`. It passed mechanical hard gates but is not evidence of coherent model quality.

## Distribution Status

The package clearly records repository MIT licensing, source-specific corpus licenses, and the absence of a standalone model-weight distribution license. Information is present, so the missing-information hard gate is satisfied; however, public redistribution remains blocked until model-weight terms are selected and approved.

The model card also records the Phase 2B.2 audit FAIL decision. No upload was attempted.
