# Experiment 013 - Dataset Quality and Splits

## Goal

This experiment improves DarkMind's dataset pipeline so training data can be tracked, deduplicated, split, and checked for eval leakage before model training.

No model training is performed by this experiment.

## Why Deduplication Matters

Exact duplicate examples can make a small model memorize repeated text instead of learning more general patterns. This is especially risky for DarkMind because the dataset is still relatively small and some generated QA examples may repeat similar lines.

The deduplication step removes exact duplicate non-empty lines globally from cleaned data while protecting short structural lines and code blocks.

## Why Train/Val/Test Splits Matter

A single merged corpus is useful for early experiments, but serious model development needs separate files:

- `train.txt` for optimization
- `val.txt` for loss checks during training
- `test.txt` for later held-out checks

The split builder works at document-block level, so `<s>...</s>` boundaries are preserved.

## Why Eval Leakage Is Dangerous

If eval prompts or expected answer snippets appear in the training data, eval results can look better than the model really is. This is a form of leakage. It does not always mean the dataset is unusable, but it must be reviewed before claiming improvement.

The leakage checker searches for exact prompt/snippet matches in the train split and warns by default. Use `--strict` when you want matches to fail the command.

## Build The Manifest

```powershell
python scripts/build_dataset_manifest.py
```

This scans `data/sources` and `data/raw_collected`, then writes:

```txt
data/dataset_manifest.jsonl
```

Known project files are marked as approved project-generated data. Unknown files are marked as `needs_review`.

## Deduplicate Cleaned Data

```powershell
python scripts/clean_text.py
python scripts/deduplicate_dataset.py
```

The default output is:

```txt
data/deduped/
```

## Build Corpus From Deduped Data

```powershell
python scripts/build_dataset_from_raw.py --input_dir data/deduped
```

This still includes `data/sources/*.txt` and uses the deduplicated data directory instead of `data/cleaned`.

## Build Train/Val/Test Splits

```powershell
python scripts/build_train_val_test.py
```

Outputs:

```txt
data/processed/splits/train.txt
data/processed/splits/val.txt
data/processed/splits/test.txt
```

## Check Eval Leakage

```powershell
python scripts/check_eval_leakage.py --eval_path data/evals/darkmind_eval_v02.jsonl
```

For stricter CI-like checks:

```powershell
python scripts/check_eval_leakage.py --strict
```

## Train With Explicit Train/Val Files

```powershell
python scripts/train_from_config.py --config configs/darkmind_30m_1000step.json --train_path data/processed/splits/train.txt --val_path data/processed/splits/val.txt
```

This bypasses the old internal train/val split and uses the explicit files.

## Current Limitations

- Deduplication is exact-line based, not semantic deduplication.
- Short repeated structure lines are intentionally preserved.
- Code blocks are protected to avoid breaking examples.
- The test split is created but not yet used by the training script.
- Leakage checks are exact substring checks and can still miss paraphrased leakage.
- Manifest metadata is heuristic and should be reviewed as the dataset grows.
