# Experiment 017 - Turkish Code Data Factory

## Goal

DarkMind should become stronger at Turkish Python and code explanations before trying to become a broad general-purpose Turkish assistant. This experiment adds a generated, project-owned coding data factory and a validation loop.

No model training is performed by this experiment.

## Why Coding First

Coding is a good first serious specialization because answers can be checked more concretely than open-ended chat. Small Python examples, common error explanations, CLI workflows, tokenizer basics, and project workflow questions give DarkMind a clearer training target.

## Generated Dataset Structure

The main generated file is:

```txt
data/raw_collected/python_examples/turkish_code_factory_v01.txt
```

Each example contains:

- `# bucket: ...`
- `# difficulty: ...`
- `Kullanıcı: ...`
- `Asistan: ...`

The factory currently generates examples across beginner Python, functions, lists, dictionaries, strings, loops, conditionals, file operations, JSON, error handling, classes, algorithms, debugging, project workflow, PyTorch basics, tokenizer basics, CLI tools, and Git basics.

## Safety Filters

The validator checks for unsafe patterns such as destructive shell commands, dangerous file deletion APIs, dynamic code execution, suspicious network posting, and credential-related wording. The goal is to keep the coding dataset educational and non-destructive.

## Syntax Validation

Python code blocks are parsed with `ast.parse`. This does not execute generated code. It only checks whether the code is syntactically valid Python.

Run:

```powershell
python scripts/validate_code_dataset_quality.py --strict
```

## Code Eval v02

The eval generator writes:

```txt
data/evals/darkmind_code_eval_v02.jsonl
```

This eval focuses on code concepts that DarkMind should answer reliably in Turkish, including append, remove, sorted, dictionary get, word count, list comprehension, common errors, classes, PyTorch, tokenizer basics, Git, and project workflow.

## Expected Limitations

- The dataset is generated and still needs human review over time.
- Syntax validation does not prove semantic correctness.
- Keyword evals are useful but shallow.
- The model may still overfit if the dataset is repeated too much.
- The factory avoids unsafe examples, so it is not intended for security exploitation or destructive system tasks.

## Next Steps

1. Run the automatic code data cycle.
2. Review dataset quality output.
3. Train tokenizer manually on `corpus_curriculum_v01.txt`.
4. Train the 30M model manually.
5. Evaluate with `darkmind_code_eval_v02.jsonl`.
6. Use eval failures to add small targeted correction examples.
