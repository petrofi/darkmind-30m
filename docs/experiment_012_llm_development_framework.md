# Experiment 012 - LLM Development Framework

## What Changed

This experiment turns DarkMind from a small training experiment into a more structured Turkish LLM development framework.

Added framework pieces:

- Experiment registry
- Experiment logging script
- Eval run comparison script
- Latest eval summary script
- Model card
- Roadmap
- Longer 30M training config
- Experimental 50M config
- RAG planning document

## Why Framework Matters

Without tracking, it is easy to confuse random changes with real improvement. DarkMind needs measurable progress:

- Which dataset was used?
- Which config was used?
- Which checkpoint was evaluated?
- Which eval file was used?
- Did pass rate improve or regress?
- Which categories are still failing?

The framework makes those questions answerable.

## How This Supports A Stronger Turkish LLM

DarkMind's path should be Turkish-first and evidence-driven. The framework connects dataset growth, coding examples, safe web collection, training configs, eval runs, and human-in-the-loop corrections into one workflow.

The goal is not to pretend the model is already strong. The goal is to create a repeatable process for getting stronger.

## Current Measurable Status

The project has:

- Turkish corpus pipeline
- Deterministic generated examples
- Python coding dataset track
- Eval files for general Turkish behavior and coding behavior
- Self-improvement candidates that require human review
- Allowlisted web data collection pipeline

No new model result is claimed by this experiment. Training and eval runs must be executed separately.

## Next Experiments

1. Rebuild `corpus_v3.txt` with the Turkish coding v02 examples.
2. Train tokenizer on `corpus_v3.txt`.
3. Train `darkmind_30m_longer_train`.
4. Run `darkmind_eval_v02` and `darkmind_code_eval_v01`.
5. Compare eval runs with `scripts/compare_eval_runs.py`.
6. Log the experiment with `scripts/log_experiment.py`.
7. Review category failures and generate correction candidates.

## Limitations

- Registry entries can still be inaccurate if filled carelessly.
- Keyword evals are useful but shallow.
- The 50M config may be too heavy and is not a guaranteed improvement.
- RAG is only planned, not implemented.
- Stronger documentation does not replace real model quality.
