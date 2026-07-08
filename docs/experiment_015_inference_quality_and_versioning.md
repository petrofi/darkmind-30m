# Experiment 015 - Inference Quality and Versioning

## Goal

This experiment improves the DarkMind inference layer without training a model. The focus is cleaner chat output, reproducible generation settings, checkpoint metadata, and tokenizer/checkpoint compatibility checks.

## Why Inference Cleanup Matters

Small language models often continue into the next dialogue turn. For example, an answer may start generating another `Kullanıcı:` block. In a chat demo this looks broken even if the first answer was usable.

The updated inference scripts trim common turn markers, remove special tokens, normalize repeated whitespace, and return a safe fallback when the model produces an empty answer.

## Common Tiny LLM Failure Modes

- Repeating the same phrase too many times
- Leaking `Kullanıcı:` or `Sen:` into the answer
- Ending in an incomplete dialogue block
- Producing an empty answer after special-token cleanup
- Mixing prompt text with the answer
- Looking worse because inference settings are too random

Repetition penalty and deterministic presets reduce these issues, but they do not make the model fundamentally smarter.

## Checkpoint And Tokenizer Compatibility

A checkpoint is only reliable with the tokenizer it was trained with. If `vocab.json` or `merges.txt` changes, token ids may no longer mean the same thing.

Use:

```powershell
python scripts/write_checkpoint_metadata.py --checkpoint checkpoints/darkmind_30m_1000step.pt --config configs/darkmind_30m_1000step.json
python scripts/check_checkpoint_compatibility.py --checkpoint checkpoints/darkmind_30m_1000step.pt
```

Metadata records tokenizer vocab size, tokenizer file hashes, dataset hash, config hash, git commit, and optional eval run path. It does not modify the checkpoint.

## Inference Presets

Presets live in:

```txt
configs/inference_presets.json
```

Current presets:

- `deterministic`: low temperature for stable demos
- `creative`: higher temperature for looser samples
- `code`: low temperature and more tokens for coding answers

## Chat Log Saving

`chat_demo.py` can save chat turns as JSONL:

```powershell
python scripts/chat_demo.py --checkpoint checkpoints/darkmind_30m_1000step.pt --save_chat_log reports/eval/chat_log.jsonl
```

Each row includes prompt, user input, answer, checkpoint path, and inference settings.

## Limitations

- Cleanup can hide formatting problems but cannot fix model knowledge.
- Repetition penalty is a sampling control, not a training improvement.
- Metadata can detect obvious tokenizer/config drift, but it cannot prove model quality.
- Smoke prompts are small and should not be treated as a full benchmark.
- Safe fallback is honest behavior for empty output, not a substitute for better training data.
