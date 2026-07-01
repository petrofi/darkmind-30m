# DarkMind Experiment Registry

This folder tracks DarkMind training and evaluation experiments.

Use `scripts/log_experiment.py` after a real experiment to append a JSONL entry to `experiments/experiment_registry.jsonl`.

The registry is for traceability only. Do not record pass rates, losses, or claims unless they were produced by an actual run and can be traced to files.

Example:

```powershell
python scripts/log_experiment.py `
  --experiment_id exp_012 `
  --name "Turkish coding track v02" `
  --eval_path data/evals/darkmind_eval_v02.jsonl `
  --eval_path data/evals/darkmind_code_eval_v01.jsonl `
  --notes "Dataset generated and syntax-validated; model not trained yet."
```
