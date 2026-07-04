# DarkMind Teacher-Student Distillation

This folder contains the teacher-student distillation pipeline for DarkMind.

Teacher-student distillation means using a stronger model to generate clean synthetic instruction-response examples, then training a smaller student model on those examples. Here, Qwen3-VL-30B is the teacher model and DarkMind-30M is the student/research model.

We are not extracting Qwen's original training data. The generator asks the local Qwen runtime to create new synthetic examples for DarkMind.

## Why This Exists

DarkMind-30M base training works technically, but v0.4 instruction tuning collapsed into identity answers. For example, identity questions were answered correctly, but unrelated prompts such as "Python nedir?" or "Docker nedir?" also received DarkMind identity answers. Therefore the next student training must start from `models/darkmind-30m-10k-step15000.pt`, not from v0.4.

The v0.1 distillation dataset starts with 2,000 examples because it is large enough to cover core software-assistant behavior while still small enough to inspect carefully before training.

## Local Qwen Server

Start LM Studio and load the local Qwen model, or use the CLI:

```powershell
& "$env:USERPROFILE\.lmstudio\bin\lms.exe" server start
& "$env:USERPROFILE\.lmstudio\bin\lms.exe" load qwen/qwen3-vl-30b --identifier local-model -y
```

The default config expects:

```text
http://localhost:1234/v1
model: local-model
```

For local overrides, copy `config.example.json` to `config.json`. The local config is ignored by git.

## Generate Dataset

Smoke test:

```powershell
python darkmind_distill/generate_qwen_distill_dataset.py --limit 30 --batch_size 1
```

Full v0.1 target:

```powershell
python darkmind_distill/generate_qwen_distill_dataset.py
```

Local 30B generation is slow. `--batch_size 1` is the safest setting for LM Studio; use larger batches only if the server stays responsive.

## Inspect Dataset

Smoke inspection:

```powershell
python darkmind_distill/inspect_distill_dataset.py --min_total 30 --skip_target_checks
```

Full inspection:

```powershell
python darkmind_distill/inspect_distill_dataset.py
```

## Train Student

The wrapper refuses to train unless inspection passes and `--confirm_train` is provided.

Preview command:

```powershell
python darkmind_distill/train_student_from_distill.py
```

Run training only after inspection passes:

```powershell
python darkmind_distill/train_student_from_distill.py --confirm_train
```

## Evaluate Student

```powershell
python darkmind_distill/eval_student_outputs.py --checkpoint models/darkmind-30m-qwen-distill-v0.1.pt --out darkmind_distill/reports/qwen_distill_v0_1_eval.md
```
