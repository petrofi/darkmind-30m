# Experiment 016 - Automated Dataset Processing

## Goal

This experiment adds an automated dataset processing engine for DarkMind. It cleans, scores, rejects, buckets, and prepares training data without approving pending review files, training a tokenizer, training a model, or touching checkpoints.

## Why Raw Data Cannot Be Trusted Directly

Raw data can contain duplicates, broken dialogue turns, web boilerplate, empty answers, repeated lines, malformed code blocks, or text that does not match the project goal. Training directly on raw or barely cleaned text can teach the model bad patterns.

DarkMind needs a pipeline where examples are measurable and reviewable before they enter the final corpus.

## Scoring Logic

`scripts/score_dataset_examples.py` walks `data/cleaned/**/*.txt`, splits content into examples, and scores each one with deterministic rules.

Positive signals include:

- Turkish text
- technical terms
- clear `Kullanıcı:` / `Asistan:` structure
- valid short Python code blocks checked with `ast.parse`
- useful answer or explanation content
- DarkMind-relevant concepts such as tokenizer, CUDA, PyTorch, Python, Transformer, checkpoint, corpus, eval, and dataset

Negative signals include:

- too short or too long examples
- broken code fences
- repeated lines
- too many URLs
- cookie/privacy/login/subscribe boilerplate
- empty answers
- answers that only repeat the question
- unfinished dialogue turns
- exact duplicate examples

## Rejected Examples

Rejected examples are written to:

```txt
data/processed_quality/rejected/rejected_dataset_v01.txt
```

They are not deleted. This makes the filtering process inspectable and reversible.

## Curriculum Dataset

`scripts/build_curriculum_dataset.py` reads:

```txt
data/processed_quality/scored/scored_dataset_v01.txt
```

It buckets examples into:

- chat basics
- identity
- fallback
- Python code
- coding errors
- LLM concepts
- data pipeline
- CUDA/GPU
- project workflow
- unknown

The final ordered corpus is written to:

```txt
data/processed/corpus_curriculum_v01.txt
```

## Limitations

- The scoring rules are heuristic, not a perfect quality detector.
- `ast.parse` checks syntax only; it does not prove code semantics.
- Some useful examples may be rejected and some weak examples may pass.
- The curriculum order is simple keyword-based bucketing.
- This pipeline does not guarantee a strong LLM without much larger, clean, diverse Turkish data.

## How This Helps DarkMind

This makes dataset growth less manual and more measurable. It gives DarkMind a safer path from raw generated/project-owned text toward a cleaner Turkish-focused training corpus.

The result is not magic. It is a better preparation layer: easier to inspect, easier to improve, and harder to accidentally pollute with poor data.
