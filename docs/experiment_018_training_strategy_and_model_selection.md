# Experiment 018 - Training Strategy and Model Selection

## Goal

This experiment adds a measurable training strategy for DarkMind. It does not claim that a new model is better. A checkpoint should only be treated as better when eval results show measurable improvement.

## Why 1000 Steps Is Only Quick Iteration

The existing 1000-step config is useful for checking that tokenizer, data, model code, and training loop work together. It is not enough evidence for a strong model. Short runs can show whether a change breaks training, but they are often too noisy for judging real quality.

## Why Longer Training May Help Or Hurt

Longer runs can improve loss and behavior when the dataset is clean and diverse. They can also overfit if the corpus is small, duplicated, or leaking eval examples. This is why longer configs now write metrics JSONL files and save both best and last checkpoints.

## Why Eval-Gated Promotion Matters

Promotion means "this checkpoint is the current best candidate." It should not happen just because training finished. The promotion script reads an eval run, computes pass rate, and refuses to copy the checkpoint unless the threshold is met.

This keeps DarkMind honest:

- no fake metrics
- no ChatGPT-level claims
- no promotion without eval evidence
- no deletion of older checkpoints

## Comparing Checkpoints

Use the benchmark suite for each candidate checkpoint:

```powershell
python scripts/benchmark_suite.py --checkpoint checkpoints/darkmind_30m_curriculum_3000step_best.pt
```

Compare:

- overall weighted pass rate
- per-eval pass rate
- categories that still fail in the individual eval JSONL files
- training metrics in `reports/training/<run_name>_metrics.jsonl`

## Choosing 30M vs 50M

The 30M configs are the default development path. They are closer to the current model size and safer for RTX 4060 Laptop GPU experimentation.

The 50M config is experimental. It uses a lower per-step batch size and gradient accumulation, but it may still be slower or too heavy depending on VRAM, thermals, and dataset size. It should only be tried after the 30M curriculum runs prove useful.

## Recommended Next Order

1. Run 30M curriculum 3000 steps.
2. Run the benchmark suite.
3. Promote only if evals improve and pass the threshold.
4. Try 30M curriculum 6000 steps.
5. Benchmark again.
6. Only then try the 50M curriculum config.

## Current Limitations

- Eval files are still small and keyword-based.
- Pass rate is useful but not a full measure of language quality.
- Longer training can overfit if the dataset remains too small.
- Promotion checks one eval run at a time.
- Human review is still needed before declaring a checkpoint meaningfully better.
