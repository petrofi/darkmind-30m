# Model Card: DarkMind-30M

## Model Name

DarkMind-30M

## Language

Turkish-focused. The project primarily targets Turkish prompts, Turkish explanations, and Turkish-first coding assistance experiments.

## Architecture

GPT-style decoder-only Transformer trained from scratch with a local tokenizer and custom training loop.

Current 30M-family configs are around 27M-30M parameters depending on tokenizer vocabulary size. Parameter count can shift when the tokenizer vocab changes because token embeddings are part of the model.

## Intended Use

- Turkish AI learning and experimentation
- Python/code explanation experiments
- Tokenizer and training pipeline education
- Small local demos
- Measuring dataset, tokenizer, and evaluation changes over time

## Not Intended For

- Medical, legal, or financial decisions
- Production advice
- Replacing professional systems
- High-stakes use
- Security-sensitive coding guidance
- Claims of ChatGPT-level general capability

## Limitations

- Tiny dataset compared with production LLMs
- Small parameter count
- High overfitting risk
- Hallucination risk
- Weak generalization
- Turkish ability is still experimental
- Coding ability is still experimental and mostly Python-focused
- Evaluation is keyword/phrase based and does not prove broad intelligence

## Data

DarkMind uses local, generated, manually approved, and source-tracked data. Web data is only intended to be used from allowlisted sources after human review.

Project policy:

- No Reddit training data
- No blind internet scraping
- No unlicensed copyrighted web text
- No automatic training on model outputs
- Human review before self-improvement examples enter training data

## Evaluation

Current eval files include:

- `data/evals/darkmind_eval_v02.jsonl`
- `data/evals/darkmind_code_eval_v01.jsonl`

Eval results should be tracked as JSONL run files and compared with `scripts/compare_eval_runs.py`.

## Safety

DarkMind uses human-in-the-loop self-improvement. Failed eval answers can produce candidate correction examples, but those candidates must be reviewed before they enter the dataset.

The project intentionally avoids blind self-training and unsafe scraping.
